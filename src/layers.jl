# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md


#---------------------------Residual Block-----------------------------------

struct ResidualBlock
    conv_layers
    norm_layers
end

ResidualBlock(chs::Int) =
   ResidualBlock((Conv((3,3), chs=>chs, pad = (1,1)), Conv((3,3), chs=>chs, pad = (1,1))), (BatchNorm(chs), BatchNorm(chs)))

function (r::ResidualBlock)(x)
    value = relu.(r.norm_layers[1](r.conv_layers[1](x)))
    r.norm_layers[2](r.conv_layers[2](value)) + x
end

#--------------------------Reflection Pad-------------------------------------

# Paper suggests using Reflection Padding. However normal padding is being used until this layer is implemented

struct ReflectionPad
    dim::Int
end

#----------------------Convolution Block--------------------------------------

ConvBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int} = (1,1), pad::Tuple{Int,Int} = (0,0)) =
    Chain(Conv(kernel, chs, stride = stride, pad = pad), ReflectionPad(kernel[1]÷2))

#-------------------------Upsample--------------------------------------------

Upsample(x) = repeat(x, inner = (2,2,1,1))

#----------------------Upsampling BLock---------------------------------------

# TODO: Use reflection padding instead of zero padding once its implemented

UpsamplingBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int}, upsample::Int, pad::Tuple{Int,Int} = (0,0)) =
    Chain(Conv(kernel, chs, stride = stride, pad = (kernel[1]÷2, kernel[2]÷2)), x -> Upsample(x))

