
require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'optim'
require 'cutorch'
require 'cunn'

local pl = require('pl.import_into')()
local printf = pl.utils.printf

local cmd = torch.CmdLine()
cmd:text()
cmd:text('A Neural Algorithm of Artistic Style')
cmd:text()
cmd:text('Options:')
cmd:option('--content', 'none', 'content image')
cmd:option('--style',         'none',   'style image')
cmd:option('--sfolder',         'none',   'style image folder')

cmd:option('--style_factor',     2e9,     'Trade-off factor between style and content')
cmd:option('--num_iters',        200,     'Number of iterations')
cmd:option('--size',             0,     'Length of image long edge (0 to use original content size)')
cmd:option('--slice',             460,     'Slice size')
cmd:option('--overlap',           75,     'overlap size')
cmd:option('--display_interval', 0,      'Iterations between image displays (0 to suppress display)')
cmd:option('--smoothness',       0,       'Total variation norm regularization strength (higher for smoother output)')
cmd:option('--init',            'image',  '{image, random}. Initialization mode for optimized image.')
cmd:option('--backend',         'cunn',   '{cunn, cudnn}. Neural network CUDA backend.')
cmd:option('--optimizer',       'lbfgs',  '{sgd, lbfgs}. Optimization algorithm.')
cmd:option('--cpu',              false,   'Optimize on CPU (only with VGG network).')
opt = cmd:parse(arg)

paths.dofile('models/util.lua')
paths.dofile('models/vgg19.lua')
paths.dofile('costs.lua')
paths.dofile('train.lua')


function slice(img, slice_size, overlap, model_to_train)
   --  assume img is in CPU
   --  assume model is either nil or a GPU model
  local height = img:size(2)
  local width = img:size()[3]

  local step = slice_size - overlap
  local i=0
  local result = {}
 
  for w =0, width, step do
    local right = math.min(width-1, w+slice_size)    
    i=i+1
    result[i] = {}
    local j=0
    for h=0, height, step do
      j=j+1
      local bottom = math.min(height-1, h+ slice_size)
      if model_to_train ~= nil then
        if right-w >=overlap and bottom-h >=overlap then
          local subimage = preprocess(image.crop(img, w,h, right, bottom),0):cuda()
          model_to_train:forward(subimage)
          collectgarbage()
        end
      else 
        local subimage = image.crop(img, w,h, right, bottom):cuda()
        local fname = 'tmp/tmp_'..j..'_'..i..'.jpg'
        image.save(fname, subimage)
        result[i][j] = fname
      end
    end
  end  
  collectgarbage()
  return result
end

function hcombine2(img1, img2, overlap)
  if img1:size()[2] ~= img2:size()[2] then
    print("Height not equal")
    return
  end

  if img2:size(3) < overlap then
    return img1
  end
  
  local height = img1:size()[2]
  local img11 = image.crop(img1,0,0, img1:size()[3]-overlap, height)
  local img1over = image.crop(img1,img1:size()[3]-overlap,0, img1:size()[3], height)
  local img2over = image.crop(img2, 0,0, overlap, height )
  local img22 = image.crop(img2, overlap, 0, img2:size()[3], height)
  
  local trans1 = torch.DoubleTensor(img1over:size())
  local ones = torch.DoubleTensor(img1over:size()):fill(1)
  local i = -1
  trans1:apply(function() i = (i + 1) % overlap; return i/(overlap-1) end)
  -- print(trans1)
  local trans2 = ones - trans1
  --image.display( img1over:map(trans2, function(xx, yy) return xx*yy end))
  --image.display(img2over:map(trans1, function(xx, yy) return xx*yy end))
  local middle = img1over:map(trans2, function(xx, yy) return xx*yy end) + img2over:map(trans1, function(xx, yy) return xx*yy end)
  --image.display(middle)
  local result = torch.cat(torch.cat(img11, middle), img22)
  return result
end


function vcombine2(img1, img2, overlap)
  if img1:size()[3] ~= img2:size()[3] then
    print("Width not equal")
    return
  end
  
  if img2:size(2) < overlap then
    return img1
  end
  if img2:size()[2] < overlap then
    return img1
  end
  local width = img1:size()[3]
  
  local img11 =    image.crop(img1,0,0, width, img1:size()[2]-overlap )
  local img1over = image.crop(img1, 0,  img1:size()[2]-overlap, width,  img1:size()[2])
  local img2over = image.crop(img2, 0,0, width, overlap )
  local img22 = image.crop(img2, 0, overlap, width, img2:size()[2])
  
  local trans1 = torch.DoubleTensor(img1over:size())
  local ones = torch.DoubleTensor(img1over:size()):fill(1)
  local i = -1
  trans1 = trans1:transpose(2,3)
  trans1:apply(function() i = (i + 1) % overlap; return i/(overlap-1) end)
  trans1 = trans1:transpose(2,3)
  local trans2 = ones - trans1
  --image.display( img1over:map(trans2, function(xx, yy) return xx*yy end))
  --image.display(img2over:map(trans1, function(xx, yy) return xx*yy end))
  local middle = img1over:map(trans2, function(xx, yy) return xx*yy end) + img2over:map(trans1, function(xx, yy) return xx*yy end)
  --image.display(middle)
  local result = torch.cat(torch.cat(img11, middle, 2), img22, 2)
  return result
end


function combine(imgs, overlap)
  local first = true
  local result
  for i=1, #imgs do
    local img = image.load(imgs[i][1])
    for j=2, #imgs[1] do
      img2= image.load(imgs[i][j])
      img = vcombine2(img, img2, overlap)
    end
    if first then
       result = img
       first = false
    else 
       result = hcombine2(result, img, overlap)
    end
    -- image.display(result)
  end
  return result
end


local means = { 104, 117, 123 }

function preprocess(img, scale)  
   -- img is 3-d with 
    local w, h = img:size(3), img:size(2)

    if scale>0 then
      local w, h = img:size(3), img:size(2)
        if w < h then
            img = image.scale(img, scale * w / h, scale)
        else
            img = image.scale(img, scale, scale * h / w)
        end
    end

    -- reverse channels
    local copy = torch.FloatTensor(img:size())
    copy[1] = img[3]
    copy[2] = img[2]
    copy[3] = img[1]
    img = copy

    img:mul(255)
    for i = 1, 3 do
        img[i]:add(-means[i])
    end
    return img:view(1, 3, img:size(2), img:size(3))
end

function depreprocess(img)
    local copy = torch.FloatTensor(3, img:size(3), img:size(4)):copy(img)
    for i = 1, 3 do
        copy[i]:add(means[i])
    end
    copy:div(255)

    -- reverse channels
    local copy2 = torch.FloatTensor(copy:size())
    copy2[1] = copy[3]
    copy2[2] = copy[2]
    copy2[3] = copy[1]
    copy2:clamp(0, 1)
    return copy2
end


-------------------------------------------------------------------
function run(prefix)
  local img = image.load(opt.content)
  if opt.size ~= nill and opt.size >0 then
    img = image.scale(img, opt.size)
  end

  imgs = slice(img, opt.slice, opt.overlap, nil)
 
  img=nill

  -- cpu_model = new_model( 'styles/renoir/the_reading.jpg', 'none', 400)
  -- cpu_model = new_model( 'none', 'styles/colors',  400)
   cpu_model = new_model( opt.style, opt.sfolder, opt.slice, style_weights)
  local k=1
  local total = #imgs * #imgs[1]
  
  --cpu_model = new_model('styles/Seurat_SundayAfternoonOnTheIslandOfGrandJatte.jpg' , 'none', opt.slice)
  for i=1, #imgs do
    for j = 1, #imgs[1] do
       print("image ", k, ' of ', total)
   --    if (i< #imgs/2 ) then
         imgs[i][j] = train(imgs[i][j],  cpu_model)
         k=k+1 
    --   end
    end
  end


  result = combine(imgs, opt.overlap)
  image.save("out/" .. prefix .. opt.content  , result)
  --image.display(result)
  --train('tardis.jpg', 'monalisa.jpg', 'none')
  --train('monalisa.jpg',  'tardis.jpg', 'none')
end



function setLayer(num, layer_list, output_layers)
    -- returns a table of bits, most significant first.
    bits = #layer_list
    local name = ""
    
    for b=bits,1,-1 do
        w=math.fmod(num,2)
        output_layers[layer_list[b]]=w
        num=(num-w)/2
        if w>0 then
          name = layer_list[b] .. '-' .. name 
        end
    end
    return name    
end


style_weights = {
        ['conv1_1'] = 1,
        ['conv2_1'] = 1,
        ['conv3_1'] = 1,
        ['conv3_2'] = 1,
        ['conv4_1'] = 1,
        ['conv5_1'] = 1,
  }
content_weights = {
        ['conv4_2'] = 1,
  }

local low_layers = {'conv1_2','conv2_2','conv3_2','conv3_3',
   'conv3_4', 'conv4_1', 'conv5_1' }

local high_layers = {'conv4_1','conv4_2','conv4_3','conv4_4','conv5_1','conv5_2',
   'conv5_3','conv5_4'}

function setLayer(num, layer_list, output_layers)
    -- returns a table of bits, most significant first.
    bits = #layer_list
    local name = ""
    for b=bits,1,-1 do
        w=math.fmod(num,2)
        output_layers[layer_list[b]]=w
        num=(num-w)/2
        if w>0 then
          name = layer_list[b] .. '-' .. name 
        end
    end    
    return name:gsub('conv','') 
end

style_weight_sum = 0
content_weight_sum = 0
for k, v in pairs(content_weights) do
    content_weight_sum = content_weight_sum + v
end
for k, v in pairs(style_weights) do
  style_weight_sum = style_weight_sum + v
end
run('uno-')


--
--for i = 0, 2^#low_layers do
--  name = setLayer(i, low_layers, style_weights)
--  print('name', name)
--  style_weight_sum = 0
--  for k, v in pairs(style_weights) do
--    style_weight_sum = style_weight_sum + v
--  end
--  if style_weight_sum == 0 then
--    print('sum 0')
--    break   
--  end
--  
--  run(name)
--end
