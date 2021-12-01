# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-----------------------------VGG19-----------------------------------

mutable struct vgg19
    slice1
    slice2
    slice3
    slice4
end


function vgg19()
    vgg = VGG19().layers[1]  # get only the first set of layers in the chain
    slice1 = Chain(vgg[1:5]...)
    slice2 = Chain(vgg[6:10]...)
    slice3 = Chain(vgg[11:15]...)
    slice4 = Chain(vgg[16:20]...)
    vgg19(slice1, slice2, slice3, slice4)
end

function (layer::vgg19)(x)
    res1 = layer.slice1(x)
    res2 = layer.slice2(res1)
    res3 = layer.slice3(res2)
    res4 = layer.slice4(res3)
    (res1, res2, res3, res4)
end
