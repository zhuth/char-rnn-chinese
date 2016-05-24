
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-noprime',false,'do not show prime text in output')
cmd:option('-length',2000,'max number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-d',0,'run as daemon (with port number)')
cmd:option('-stop','\n\n\n\n\n','stop sampling when detected')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
torch.manualSeed(opt.seed)

function load(opt)
    -- load the model checkpoint
    if not lfs.attributes(opt.model, 'mode') then
        gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
    end
    local checkpoint = torch.load(opt.model)
    protos = checkpoint.protos
    protos.rnn:evaluate() -- put in eval mode so that dropout works properly

    -- initialize the vocabulary (and its inverted version)
    local vocab = checkpoint.vocab
    local ivocab = {}
    for c,i in pairs(vocab) do ivocab[i] = c end

    -- initialize the rnn state to all zeros
    gprint('creating an LSTM...')
    local current_state
    local num_layers = checkpoint.opt.num_layers
    current_state = {}
    for L = 1,checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
        if opt.gpuid >= 0 then h_init = h_init:cuda() end
        table.insert(current_state, h_init:clone())
        table.insert(current_state, h_init:clone())
    end
    state_size = #current_state
    
    return {
        vocab = checkpoint.vocab,
        ivocab = ivocab,
        protos = protos,
        current_state = current_state,
        prediction = nil
    }
end

-- parse characters from a string
function get_char(str)
    local len  = #str
    local left = 0
    local arr  = {0, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc}
    local unordered = {}
    local start = 1
    local wordLen = 0
    while len ~= left do
        local tmp = string.byte(str, start)
        local i   = #arr
        while arr[i] do
            if tmp >= arr[i] then
                break
            end
            i = i - 1
        end
        wordLen = i + wordLen
        local tmpString = string.sub(str, start, wordLen)
        start = start + i
        left = left + i
        unordered[#unordered+1] = tmpString
    end
    return unordered
end

function prime(model, opt)
    -- do a few seeded timesteps    
    local seed_text = opt.primetext
    if string.len(seed_text) > 0 then
        gprint('seeding with ' .. seed_text)
        gprint('--------------------------')
        local chars = get_char(seed_text)
        for i,c in ipairs(chars) do
            if model.vocab[c] == nil then c = 'UNKNOW' end
            prev_char = torch.Tensor{model.vocab[c]}
            if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
            local lst = model.protos.rnn:forward{prev_char, unpack(model.current_state)}
            -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
            model.current_state = {}
            for i=1,state_size do table.insert(model.current_state, lst[i]) end
            model.prediction = lst[#lst] -- last element holds the log probabilities
        end
    else
    -- fill with uniform probabilities over characters (? hmm)
        gprint('missing seed text, using uniform probability over first character')
        gprint('--------------------------')
        model.prediction = torch.Tensor(1, #model.ivocab):fill(1)/(#model.ivocab)
        if opt.gpuid >= 0 then model.prediction = model.prediction:cuda() end
    end
end

function sample(model, opt)
    -- start sampling/argmaxing
    result = ''
    confidence = 0
    for i=1, opt.length do

        -- log probabilities from the previous timestep
        -- make sure the output char is not UNKNOW
        if opt.sample == 0 then
            -- use argmax
            local _, prev_char_ = model.prediction:max(2)
            confidence = confidence + math.log(_:resize(1))
            prev_char = prev_char_:resize(1)
        else
            -- use sampling
            -- real_char = 'UNKNOW'
            -- while(real_char == 'UNKNOW') do
            model.prediction:div(opt.temperature) -- scale by temperature
            local probs = torch.exp(model.prediction):squeeze()
            probs:div(torch.sum(probs)) -- renormalize so probs sum to one
            prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
            confidence = confidence + math.log(probs[prev_char[1]])
            real_char = model.ivocab[prev_char[1]]
            -- end
        end

        -- forward the rnn for next character
        local lst = model.protos.rnn:forward{prev_char, unpack(model.current_state)}
        model.current_state = {}
        for i=1,state_size do table.insert(model.current_state, lst[i]) end
        model.prediction = lst[#lst] -- last element holds the log probabilities

        -- io.write(model.ivocab[prev_char[1]])
        if real_char == 'UNKNOW' then real_char = '□' end
        result = result .. real_char

        -- in my data, five \n represent the end of each document
        -- so count \n to stop sampling
        if string.find(result, opt.stop) then break end
    end
    return {
        output = result,
        primetext = opt.primetext,
        confidence = confidence / opt.length
    }
end

if opt.d > 0 then
    
    models = {}
    
    xavante = require("xavante")
    xavante.filehandler = require("xavante.filehandler")
    xavante.redirecthandler = require("xavante.redirecthandler")
    
    wsapi = require("wsapi")
    wsapi.xavante = require("wsapi.xavante")
    wsapi.request = require("wsapi.request")

    json = require ('json')

    -- JSON RPC Server main
    local function serve(req)
        local dup = {}
        for orig_key, orig_value in pairs(opt) do
            dup[orig_key] = orig_value
        end
        for k, v in pairs(json.decode(req.POST.post_data)) do
            dup[k] = v
        end

		if opt.gpuid >= 0 then
	        cutorch.manualSeed(dup.seed)
	    else
			torch.manualSeed(dup.seed)
		end
		
        if models[dup.model] == nil then models[dup.model] = load(dup) end
        model = models[dup.model]
        prime(model, dup)
        local jsonResponse = sample(model, dup)
        jsonResponse['model'] = dup.model
		
        return json.encode( jsonResponse ) 
    end

    ---  WSAPI handler
    -- @param wsapi_env WSAPI environment
    function wsapi_handler(wsapi_env)
       local headers = { ["Content-type"] = "application/json" }
       local req = wsapi.request.new(wsapi_env)

       local r = serve(req)
       headers["Content-length"] = tostring(#r)

       local function xmlrpc_reply(wsapienv)
          coroutine.yield(r)
       end

       return 200, headers, coroutine.wrap(xmlrpc_reply)
    end
    
    function wsapi_listmodels(wsapi_env)
       local headers = { ["Content-type"] = "application/json" }
       local req = wsapi.request.new(wsapi_env)
       
       models = {}
       for file in lfs.dir[[cv/]] do
           if string.sub(''..file, -3) == '.t7' then
               models[#models+1] = 'cv/' .. file
           end
       end
       
       
       local r = json.encode( { models = models } )
       print("r="..r)
       headers["Content-length"] = tostring(#r)

       local function xmlrpc_reply(wsapienv)
          coroutine.yield(r)
       end

       return 200, headers, coroutine.wrap(xmlrpc_reply)
    end

    local rules = {
        { match = "^/$", with = xavante.redirecthandler, params = {"index.html"} }, 
        { match = "^/rpc$", with = wsapi.xavante.makeHandler(wsapi_handler) },
        { match = "^/models$", with = wsapi.xavante.makeHandler(wsapi_listmodels) },
        { match = ".", with = xavante.filehandler, params = {baseDir = './static/'} } 
    }
    
    local config = { server = {host = "*", port = opt.d}, defaultHost = { rules = rules} }

    xavante.HTTP(config)
    xavante.start()
    
else
    local model = load(opt)
    prime(model, opt)
    v = sample(model, opt)
    print('Confidence: ' .. v.confidence .. '\n--------------------------')
    if not opt.noprime then io.write(v.primetext) end
    print(v.output .. '\n')
end
