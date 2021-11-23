__precompile__()

module FastStyleTransfer

using Flux, Metalhead, Images, CUDA, NNlib, BSON

export train, stylize

include("utils.jl")
include("layers.jl")
include("vgg.jl")
include("transformer_net.jl")
include("neural_style.jl")

end # module
