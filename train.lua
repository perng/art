
local pl = require('pl.import_into')()
local printf = pl.utils.printf

local  style_weights = {
        ['conv1_1'] = 1,
        ['conv2_1'] = 1,
        ['conv3_1'] = 1,
        ['conv4_1'] = 1,
        ['conv5_1'] = 1,
  }
local  content_weights = {
        ['conv4_2'] = 1,
  }

  local style_weight_sum = 0
  local content_weight_sum = 0
  for k, v in pairs(style_weights) do
    style_weight_sum = style_weight_sum + v
  end

  for k, v in pairs(content_weights) do
    content_weight_sum = content_weight_sum + v
  end


function opfunc_closure(model)
  local wmodel = model
  return  function (input)
    -- forward prop
    wmodel:forward(input)

    -- backpropagate
    local loss = 0
    local grad = opt.cpu and torch.FloatTensor() or torch.CudaTensor()
    grad:resize(wmodel.output:size()):zero()
    for i = #wmodel.modules, 1, -1 do
        local module_input = (i == 1) and input or wmodel.modules[i - 1].output
        local module = wmodel.modules[i]
        local name = module._name

        -- add content gradient
        if name and content_weights[name] then
            local c_loss, c_grad = content_grad(module.output, img_activations[name])
            local w = content_weights[name] / content_weight_sum
            --printf('[content]\t%s\t%.2e\n', name, w * c_loss)
            loss = loss + w * c_loss
            grad:add(w, c_grad)
        end

        -- add style gradient
        if name and style_weights[name] then
            local s_loss, s_grad = style_grad(module.output, art_grams[name])
            local w = opt.style_factor * style_weights[name] / style_weight_sum
            --printf('[style]\t%s\t%.2e\n', name, w * s_loss)
            loss = loss + w * s_loss
            grad:add(w, s_grad)
        end
        grad = module:backward(module_input, grad)
    end

    -- total variation regularization for denoising
    grad:add(total_var_grad(input):mul(opt.smoothness))
    
    return loss, grad:view(-1)
end

end

function get_input(img)
-- image to optimize
  local input
  if opt.init == 'image' then
    input = img
  elseif opt.init == 'random' then
    input = preprocess(
        torch.randn(3, img:size(3), img:size(4)):mul(0.1):add(0.5):clamp(0, 1)
    )
    input = input:cuda()
  else
    error('unrecognized initialization option: ' .. opt.init)
  end
  return input
end

function iterate(xmodel, input, content_image)
  local timer = torch.Timer()
  --local output = depreprocess(input):double()
  --if opt.display_interval > 0 then
  --    image.display(output)
  --end

  -- make directory to save intermediate frames
  -- local frames_dir = 'out-'  .. content_image
  --if not paths.dirp(frames_dir) then
  --  paths.mkdir(frames_dir)
  --end
  --image.save(paths.concat(frames_dir, '0.jpg'), output)

  -- set optimizer options
  local optim_state
  if opt.optimizer == 'sgd' then
      optim_state = {
          momentum = 0.9,
          dampening = 0.0,
      }
      optim_state.learningRate = 1e-3
    
  elseif opt.optimizer == 'lbfgs' then
    optim_state = {
        maxIter = 3,
        learningRate = 1,
    }
  else
    error('unknown optimizer: ' .. opt.optimizer)
  end

  -- optimize
  for i = 1, opt.num_iters do
  
    -- local _, loss = optim[opt.optimizer](opfunc_closure(xmodel), input, optim_state)
    local _, loss = optim[opt.optimizer](opfunc_closure(xmodel), input, optim_state)
    loss = loss[1]

    -- anneal learning rate
    if opt.optimizer == 'sgd' and i % 100 == 0 then
        optim_state.learningRate = 0.75 * optim_state.learningRate
    end

    if i % 10 == 0 then
        printf('iter %5d\tloss %8.2e\tlr %8.2e\ttime %4.1f\n',
            i, loss, optim_state.learningRate, timer:time().real)
    end

    if i % 20 == 0 then
        if opt.display_interval > 0 and i % opt.display_interval == 0 then
            output = depreprocess(input):double()
            image.display(output)
        end
        -- image.save(paths.concat(frames_dir, i .. '.jpg'), output)
    end
  end

  output = depreprocess(input)
  return output
end
------------------------------------------
function load_content_image(gpu_model,content_weights, content_image, scale )
  local img = preprocess(content_image, 0):cuda()
  gpu_model:forward(img)
  local img_activations, _ = collect_activations(gpu_model, content_weights, {})
  return img, img_activations
end
-------------------------------------------
function rescale(img, scale)

    local w, h = img:size(3), img:size(2)
    if scale then
        if w < h then
            img = image.scale(img, scale * w / h, scale)
        else
            img = image.scale(img, scale, scale * h / w)
        end
    end
    return img
end

------------------------
function load_style_images(cpu_model, style_weights, style_image, 
               style_folder, slice_size)
  gpu_model = cpu_model:cuda()
  if style_folder ~= "none" then 
   if string.sub(style_folder, -1) ~= '/' then
      style_folder = style_folder ..'/'
   end
   style_folder = string.gsub(style_folder, " ", "\\ ")
   for f in io.popen("ls ".. style_folder ):lines() do
     f = string.gsub(f, " ", "\\ ")
     print("Load style file " .. style_folder..f)
     local art = preprocess( image.load(style_folder..f), slice_size)
     if not opt.cpu then
         art = art:cuda()
     end
     --slice(art, slice_size, slice_size/5, gpu_model)
     gpu_model:forward(art)
     art = nil
     collectgarbage()
   end
  end
  if style_image ~= "none" then 
    local art = preprocess( image.load(style_image), slice_size)
    if not opt.cpu then
        art = art:cuda()
    end
    -- slice(art, slice_size, slice_size/5, gpu_model)
    gpu_model:forward(art)
    art = nil
  end
  local _, art_grams = collect_activations(gpu_model, {}, style_weights)
  collectgarbage()
  return gpu_model, art_grams
end
----------------------------------------------------------------
function new_model(style_image, style_folder, scale)
  -- load vgg19 model 
  local vgg_path = 'models/vgg_normalized.th'
  if not paths.filep(vgg_path) then
         print('ERROR: could not find VGG model weights at ' .. vgg_path)
         print('run download_models.sh to download model weights')
         error('')
  end
  local cpu_model = create_vgg(vgg_path, opt.backend)
  local gpu_model = cpu_model:cuda()
  gpu_model, art_grams = load_style_images(gpu_model, style_weights, 
                                           style_image, style_folder, scale)
  cpu_model = gpu_model:float()
  gpu_model = nil 
  collectgarbage()
  return cpu_model
end

------------------------

function train(content_image,  cpu_model)
  content_image_gpu = image.load(content_image):cuda()
  gpu_model = cpu_model:cuda()
  
  img, img_activations = load_content_image(gpu_model,content_weights , content_image_gpu)
  input = get_input(img)
  output = iterate(gpu_model, input, content_image_gpu)
  -- image.save('final-' .. content_image, output)
  cpu_model = nil 
  gpu_model = nil 
  img = nil 
  output:float()
  collectgarbage()
  local fname = 'final-' .. content_image
  image.save(fname, output)
  return fname
end

 --train(opt.content, opt.style, opt.sfolder)
