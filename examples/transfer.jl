
using Flux, Metalhead, Images, CUDA, NNlib, BSON, Random

CUDA.allowscalar(false)

include("..\\src\\utils.jl")
include("..\\src\\layers.jl")
include("..\\src\\vgg.jl")
include("..\\src\\transformer_net.jl")
include("..\\src\\neural_style.jl")

training_data = "C:\\Users\\michael\\Desktop\\mscoco\\train2017\\train2017"
style_folder = "C:\\Users\\michael\\Desktop\\Julia\\FastStyleTransfer.jl\\images"
style_image = "georges-lemmen_the-beach-at-heist_1891.jpg"
style_path = joinpath(style_folder, style_image)

model_path = "C:\\Users\\michael\\Desktop\\Julia\\FastStyleTransfer.jl\\models"
batch_size = 128
η = 0.01
epochs = 1

content_weight = 1.0
style_weight = 1.0

nimages = 100

train(training_data, batch_size, η, style_path, epochs, model_path, content_weight, style_weight, TransformerNet; images = nimages)
