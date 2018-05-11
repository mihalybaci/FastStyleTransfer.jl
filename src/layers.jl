# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

# TODO: Use moving mean and std at test time
mutable struct InstanceNorm <: layers
    β
    γ
    ϵ
end

Flux.treelike(InstanceNorm)

InstanceNorm(chs::Int; initβ = zeros, initγ = ones, ϵ = 1.0e-8) = InstanceNorm(param(initβ(chs)), param(initγ(chs)), ϵ)

(IN::InstanceNorm)(x) = reshape(IN.γ, (1,1,IN.chs,1) .* ((x .- mean(x, 4)) ./ std(x, 4)) .+ reshape(IN.β, (1,1,IN.chs,1))

#----------------------------------------------------------------------------------------------------------------

mutable struct ResidualBlock <: block
    conv_layers
    norm_layers
end

Flux.treelike(ResidualBlock)

function ResidualBlock(chs::Int)
    ResidualBlock((Conv((3,3), chs=>chs, pad = (1,1)), Conv((3,3), chs=>chs, pad = (1,1))), (InstanceNorm(chs), InstanceNorm(chs)))
end

function (r::ResidualBlock)(x)
    value = relu.(r.norm_layers[1](r.conv_layers[1](x)))
    r.norm_layers[2](r.conv_layers[2](value)) + x
end

#----------------------------------------------------------------------------------------------------------------

mutable struct ReflectionPad <: layers
    dim::Int
end

Flux.treelike(ReflectionPad)

#----------------------------------------------------------------------------------------------------------------

mutable struct ConvBlock <: block
    pad
    conv
end

Flux.treelike(ConvBlock)

function ConvBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int} = (1,1), pad::Tuple{Int,Int} = (0,0))
    ConvBlock(ReflectionPad(kernel[1]÷2), Conv(kernel, chs, stride = stride, pad = pad))
end

function (c::ConvBlock)(x)
    c.conv(c.pad(x))
end

#----------------------------------------------------------------------------------------------------------------

mutable struct Upsample{T} <: layers
    scale_factor::T
end

Flux.treelike(Upsample)

#----------------------------------------------------------------------------------------------------------------

mutable struct UpsamplingBlock <: block
    upsample
    pad
    conv
end

Flux.treelike(UpsamplingBlock)

function UpsamplingBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int}, upsample::Int, pad::Tuple{Int,Int} = (0,0))
    UpsamplingBlock(Upsample(upsample), ReflectionPad(kernel[1]÷2), Conv(kernel, chs, stride = stride, pad = pad))
end

function (u::UpsamplingBlock)(x)
    u.conv(u.pad(u.upsample(x)))
end
