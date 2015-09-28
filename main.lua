
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
cmd:option('--slice_size',  400,  'slice size')
cmd:option('--overlap_size',  50,  'overlap size')
cmd:option('--content',         'none',   'Path to content image')
cmd:option('--style_factor',     2e9,     'Trade-off factor between style and content')
cmd:option('--num_iters',        80,     'Number of iterations')
cmd:option('--size',             500,     'Length of image long edge (0 to use original content size)')
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
  print("slicing for style: ", model_to_train ~= nil)
  local height = img:size()[2]
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
      print('slicing w,h,right,bottom', w, h,right,bottom)
      if model_to_train ~= nil then
        if right-w >=slice_size and bottom-h >=slice_size then
          local subimage = image.crop(img, w,h, right, bottom)
          model_to_train:forward(subimage)
          collectgarbage()
        end
      else 
        local subimage = image.crop(img, w,h, right, bottom)
        fname = 'tmp/tmp_'..j..'_'..i..'.jpg'
        image.save(fname, subimage)
        result[i][j] = fname
      end
    end
  end  
  collectgarbage()
  return result
end

function hcombine2(img1, img2, overlap)
  print('hcombine')
  print ("img1", img1:size())
  print ("img2", img2:size())
  if img1:size()[2] ~= img2:size()[2] then
    print("Height not equal")
    return
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
  print('vcombine')
  if img1:size()[3] ~= img2:size()[3] then
    print("Width not equal")
    return
  end
  
  print('img1:', img1:size())
  print('img2:', img2:size())
  
  if img2:size()[2] < overlap then
    return img1
  end
  local width = img1:size()[3]
  print('img1:', img1:size())
  print(" width, img1:size()[2]-overlap ",  width, img1:size()[2]-overlap )
  
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
    print('imgs[i][1]', imgs[i][1])
    local img = image.load(imgs[i][1])
    print("img:", img:size())
    for j=2, #imgs[1] do
      print('imgs[i][j]', imgs[i][j])
      img2= image.load(imgs[i][j])
      print("img2:", img2:size())
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
    local copy = torch.CudaTensor(img:size())
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


function mainx()
  --local img = image.load('smallmonalisa.jpg')
  --slice(img)
q2=image.load('tardis_160.jpg')  
q1=image.load('munch.jpg')
q1=image.scale(q1, q2:size()[3], q2:size()[2])
--itorch.image(q1)
--itorch.image(q2)
local overlap = 100 
g1 = image.crop(q2,0,0,600,546 )
image.save("g1.jpg", g1)
--itorch.image(g1)
g2 = image.crop(q1,370,0, 970, 546 )
image.save("g2.jpg", g2)
--itorch.image(g2)
  local result= hcombine2(g1,g2, 230)
  print("result", result:size())
  
end

-------------------------------------------------------------------
function main()
  local img = image.load('small_mona_lisa.jpg')
  imgs = slice(img, 400, 50)
  img=nill

  cpu_model = new_model( 'none', 'styles/glass_art', 400)
  local k=1
  local total = #imgs * #imgs[1]
  
  for i=1, #imgs do
    for j = 1, #imgs[1] do
       print("image ", k, ' of ', total)
       k=k+1 
       imgs[i][j] = train(imgs[i][j],  cpu_model)
      --if (i < #imgs/2) then
      -- imgs[i][j] = train(imgs[i][j],  "styles/starry_night.jpg", 'none', 400)
      --else 
      -- imgs[i][j] = train(imgs[i][j],  "styles/the_scream.jpg", 'none', 400)
      --end
    end
  end
  result = combine(imgs, 50)
  image.save("out.jpg", result)
  image.display(result)
  --train('tardis.jpg', 'monalisa.jpg', 'none')
  --train('monalisa.jpg',  'tardis.jpg', 'none')
end

main()
