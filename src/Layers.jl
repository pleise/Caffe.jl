####################################
#### Layer types and functions #####
####################################


type ConvLayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    bias_term::Bool # whether have a bias or not #optional in caffe
    
    lr_mult_filter::Int64
    decay_mult_filter::Int64
    lr_mult_bias::Int64
    decay_mult_bias::Int64
    
    #Convolution parameters:
    num_output::Int64
    
    kernel_size::Int64 # rectengular kernel
    kernel_size_h::Int64
    kernel_size_w::Int64
    
    stride::Int64 #optinoal in caffe
    stride_h::Int64  #optinoal in caffe
    stride_w::Int64  #optinoal in caffe
    
    
    pad::Int64  #optinoal in caffe
    pad_h::Int64  #optinoal in caffe
    pad_w::Int64  #optinoal in caffe
    
    weight_filler_type::ASCIIString
    std_gausian_filler::Float64
    bias_filler_type::ASCIIString
    bias_filler_value::Float64
    
    phase::ASCIIString
end


function addConvLayer!(n ;name="", bottomLayer="", topLayer="",  bias_term=true, lr_mult_filter=0, decay_mult_filter=0,
    lr_mult_bias=0, decay_mult_bias=0, num_output=0, kernel=0, kernel_h=0, kernel_w=0, stride=0, stride_h=0, 
    stride_w=0, pad=0, pad_h=0, pad_w=0, weight_filler_type="", std_gaussian_filler=0, bias_filler_type="", 
    bias_filler_value=0, phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=ConvLayer(name, bottomLayer, topLayer, bias_term, lr_mult_filter, decay_mult_filter,
    lr_mult_bias, decay_mult_bias, num_output, kernel, kernel_h, kernel_w, stride, stride_h, 
    stride_w, pad, pad_h, pad_w, weight_filler_type, std_gaussian_filler, bias_filler_type, 
    bias_filler_value, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end

type PoolingLayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    kernel::Int64
    kernel_h::Int64
    kernel_w::Int64
    
    #optinoal in caffe
    pool_type::ASCIIString #MAX (default), AVE, STOCHASTIC
    
    pad::Int64
    pad_h::Int64
    pad_w::Int64
    
    stride::Int64
    stride_h::Int64
    stride_w::Int64
   
    phase::ASCIIString
 
end

function addPoolingLayer!(n ; name="", bottom="", top="", kernel=0, kernel_h=0, kernel_w=0, pool_type="", pad=0, pad_h=0, pad_w=0,
    strid=0, stride_h=0, stride_w=0, phase="")
    
    p=PoolingLayer(name, bottom, top, kernel, kernel_h, kernel_w, pool_type, pad, pad_h, pad_w,
    strid, stride_h, stride_w, phase)
    
    b=NameInUse(n, p.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if p.name=="" || p.bottom=="" || p.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, p)
    
    #return p
    
    
end


type InnerProductLayer <: AbstractLayer
    
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    #Convolution parameters:
    num_output::Int64
    
    
    bias_term::Bool # whether have a bias or not #optional in caffe
    
    lr_mult_filter::Int64
    decay_mult_filter::Int64
    lr_mult_bias::Int64
    decay_mult_bias::Int64
    
    
    
    weight_filler_type::ASCIIString #strongly recommended to use "constant" (default) or "gaussian"
    std_gausian_filler::Float64
    bias_filler_type::ASCIIString
    bias_filler_value::Float64
    
    phase::ASCIIString
    
    
end


function addIPLayer!(n ;name="", bottomLayer="", topLayer="",  num_output=0, bias_term=true, lr_mult_filter=0, decay_mult_filter=0,
    lr_mult_bias=0, decay_mult_bias=0, weight_filler_type="", std_gaussian_filler=0, bias_filler_type="", 
    bias_filler_value=0, phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=InnerProductLayer(name, bottomLayer, topLayer, num_output, bias_term, lr_mult_filter, decay_mult_filter,
    lr_mult_bias, decay_mult_bias, weight_filler_type, std_gaussian_filler, bias_filler_type, 
    bias_filler_value, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end

type SoftMaxLayer <: AbstractLayer
    name::ASCIIString
    bottom::Array{ASCIIString, 1}
    top::Array{ASCIIString, 1}
    
    phase::ASCIIString
    
end

function addSoftMaxLayer!(n ;name="", bottomLayer=[""], topLayer=[""], phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=SoftMaxLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" #|| c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end

type ReLULayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    phase::ASCIIString
   
end


function addReLULayer!(n ;name="", bottomLayer="", topLayer="", phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=ReLULayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end


type SoftmaxWithLossLayer <: AbstractLayer
    name::ASCIIString
    bottom::Array{ASCIIString, 1}
    top::Array{ASCIIString, 1}   
    
    phase::ASCIIString
end


function addSoftmaxWithLossLayer!(n ;name="", bottomLayer=[""], topLayer=[""], phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=SoftmaxWithLossLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    
    if c.name=="" #|| c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return c
    
end


type HingeLossLayer <: AbstractLayer
    name::ASCIIString
    bottom::Array{ASCIIString, 1}
    top::Array{ASCIIString, 1}  
    norm::ASCIIString # L1 or L2
    
    phase::ASCIIString
end


function addHingeLossLayer!(n ;name="", bottomLayer=[""], topLayer=[""], hinge_norm="L1", phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=HingeLossLayer(name, bottomLayer, topLayer, hinge_norm, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    
    if c.name=="" #|| c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return c
    
end


type SigmoidCrossEntropyLossLayer <: AbstractLayer
    name::ASCIIString
    bottom::Array{ASCIIString, 1}
    top::Array{ASCIIString, 1} 
    
    phase::ASCIIString
end


function addSigmoidCrossEntropyLossLayer!(n ;name="", bottomLayer=[""], topLayer=[""], phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=SigmoidCrossEntropyLossLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    
    if c.name=="" #|| c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return c
    
end


type InfogainLossLayer <: AbstractLayer
    name::ASCIIString
    bottom::Array{ASCIIString, 1}
    top::Array{ASCIIString, 1}   
    
    phase::ASCIIString
end


function addInfogainLossLayer!(n ;name="", bottomLayer=[""], topLayer=[""], phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=InfogainLossLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    
    if c.name=="" #|| c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return c
    
end


type AccuracyLayer <: AbstractLayer
    name::ASCIIString
    bottom::Array{ASCIIString, 1}
    top::Array{ASCIIString, 1}   
    phase::ASCIIString
    
end


function addAccuracyLayer!(n ;name="", bottomLayer=[""], topLayer=[""], phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=AccuracyLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    
    if c.name=="" #|| c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return c
    
end


type SigmoidLayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    phase::ASCIIString
end


function addSigmoidLayer!(n ;name="", bottomLayer="", topLayer="", phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=SigmoidLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end


type TanHLayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    phase::ASCIIString
end


function addTanHLayer!(n ;name="", bottomLayer="", topLayer="", phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=TanHLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end

type AbsValLayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    phase::ASCIIString
end



function addAbsValLayer!(n ;name="", bottomLayer="", topLayer="", phase="")
    
    #Inztiantiation of the ConvLayer-object
    
    c=AbsValLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end


type BNLLLayer <: AbstractLayer
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    phase::ASCIIString
end


function addBNLLLayer!(n ;name="", bottomLayer="", topLayer="", phase="")
    #Binominal Log Likelihood Layer
    
    #Inztiantiation of the ConvLayer-object
    
    c=BNLLLayer(name, bottomLayer, topLayer, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end



type PowerLayer <: AbstractLayer
    
    name::ASCIIString
    bottom::ASCIIString
    top::ASCIIString
    
    #output=(shift + scale * input) ^ power 
    power::Float64
    scale::Float64
    shift::Float64
    
    phase::ASCIIString
    
end


function addPowerLayer!(n ;name="", bottomLayer="", topLayer="", power=0.0, scale=0.0, shift=0.0, phase="")
    #Binominal Log Likelihood Layer
    
    #Inztiantiation of the ConvLayer-object
    
    c=PowerLayer(name, bottomLayer, topLayer, power, scale, shift, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.bottom=="" || c.top==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end



type HDF5DataLayer <: AbstractLayer
    name::ASCIIString
    top::Array{ASCIIString, 1}
    
    source::Array{ASCIIString, 1}
    batch_size::Int64
    
    phase::ASCIIString
    
    
end


function addHDF5DataLayer!(n ;name="", topLayer=[""], source=[""], batch_size=0, phase="")
    #Binominal Log Likelihood Layer
    
    #Inztiantiation of the ConvLayer-object
    
    c=HDF5DataLayer(name, topLayer, source, batch_size, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.top=="" || c.source==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end


type DataBaseLayer <: AbstractLayer
    name::ASCIIString
    top::Array{ASCIIString, 1}
    
    source::Array{ASCIIString, 1}
    batch_size::Int64
    
    rand_skip::Int64
    backend::ASCIIString
    
    phase::ASCIIString
    scale::Float64
       
end


function addDataBaseLayer!(n ;name="", top=[""], source=[""], batch_size=0, rand_skip=0, backend="LMDB", phase="", scale=0.0)
    #Binominal Log Likelihood Layer
    
    #Inztiantiation of the ConvLayer-object
    
    c=DataBaseLayer(name, top, source, batch_size, rand_skip, backend, phase, scale)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end


type ImageDataLayer <: AbstractLayer
    name::ASCIIString
    top::Array{ASCIIString, 1}
    
    source::Array{ASCIIString, 1}
    batch_size::Int64
    
    rand_skip::Int64
    shuffle::Bool
    
    new_height::Int64
    new_width::Int64
    
    phase::ASCIIString
       
end

function addImageDataLayer!(n ;name="", topLayer=[""], source=[""], batch_size=0, rand_skip=0, shuffle=false, 
    new_height=-1, new_width=-1, phase="")
    #Binominal Log Likelihood Layer
    
    #Inztiantiation of the ConvLayer-object
    
    c=ImageDataLayer(name, topLayer, source, batch_size, rand_skip, shuffle, new_height, new_width, phase)
    
    b=NameInUse(n, c.name)
    if b
        error("Layername already in use! Choose another name!")
    end
    
    if c.name=="" || c.top=="" || c.source==""
        error("Please initialize the Layer properly, with its relevant properties!")
    end
    
    # Add this object to the neural-net-object n
    push!(n.Layers, c)
    
    #return n
    
end



function removeLayer!(n::NeuralNet, name::ASCIIString)
    k=0
    k2=0
    for i in n.Layers
        if k==0
            k2+=1 # rownumber of the searched Layer
        end
        if i.name==name
            #deleteat!(i)
            println("Following layer removed: $i")
            k=1
        end  
    end
    
    if k==0
        println("Layer not found! Maybe it is already deleted.")
        return
    end
    
    deleteat!(n.Layers, k2)
    
end


###########################################
####      writeLayer2File methods      ####
###########################################


function writeLayer2File(f, L::HDF5DataLayer)
    # HDF5 DataLayer
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"HDF5Data\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    write(f, "\n")
    write(f, "  hdf5_data_param \{\n")
    for j=1:length(L.source)
        write(f, string("    source: \"", L.source[j], "\"\n"))
    end
    write(f, string("    batch_size: ", L.batch_size, "\n"))
    write(f, "    \}\n")
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::ReLULayer)
   
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"ReLU\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end
    write(f, string("  bottom: \"", L.bottom, "\"\n"))
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end

function writeLayer2File(f, L::PoolingLayer)
    # HDF5 DataLayer
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Pooling\"\n")
    
    write(f, string("  top: \"", L.top, "\"\n"))
    write(f, string("  bottom: \"", L.bottom, "\"\n"))
    
    write(f, "  pooling_param \{\n")
    
    write(f, string("    pool: ", L.pool_type, "\n"))
    if L.kernel > 0
        write(f, string("    kernel_size: ", L.kernel, "\n"))
    elseif L.kernel_w >0 && L.kernel_h >0
        write(f, string("    kernel_w: ", L.kernel_w, "\n"))
        write(f, string("    kernel_h: ", L.kernel_h, "\n"))
    else
        error("Kernelsize in the pooling layer $L not defined properly!")
    end
    
    if L.pad >0
        write(f, string("    pad: ", L.pad, "\n"))
    elseif L.pad_w>0 && L.pad_h >0
        write(f, string("    pad_w: ", L.pad_w, "\n"))
        write(f, string("    pad_h: ", L.pad_h, "\n"))
    end
    
    if L.stride >0
        write(f, string("    stride: ", L.stride, "\n"))
    elseif L.pad_w>0 && L.pad_h >0
        write(f, string("    stride_w: ", L.stride_w, "\n"))
        write(f, string("    stride_h: ", L.stride_h, "\n"))
    else
        write(f, string("    stride: 1\n"))
    end
        
    
    write(f, "  \}\n")
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end



function writeLayer2File(f, L::SoftMaxLayer)
    
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Softmax\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom[j], "\"\n"))
    end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::InnerProductLayer)
    
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"InnerProduct\"\n")
   
    write(f, string("  top: \"", L.top, "\"\n"))
    write(f, string("  bottom: \"", L.bottom, "\"\n"))
    write(f, "  param \{")
    write(f, string("  lr_mult: ", L.lr_mult_filter, " decay_mult: ", L.decay_mult_filter))
    write(f, "  \}\n")
    write(f, "  param \{")
    write(f, string("  lr_mult: ", L.lr_mult_bias, " decay_mult: ", L.decay_mult_filter))
    write(f, "  \}\n")
    write(f, "  inner_product_param \{\n")
    write(f, string("    num_output: ", L.num_output, "\n"))
    write(f, string("    weight_filler \{ \n"))
    write(f, string("      type: \"", L.weight_filler_type, "\"\n"))
    if L.weight_filler_type == "gaussian"
        write(f, string("      std: ", L.std_gausian_filler, "\n"))
    end
    write(f, "  \}\n")
    write(f, string("  bias_filler \{ \n"))
    write(f, string("    type: \"", L.bias_filler_type, "\"\n"))
    if L.bias_filler_type == "constant"
        write(f, string("    value: ", L.bias_filler_value, "\n")) # EVTL ALS INT64 Deklarieren!!!
    end
    write(f, "    \}\n")
    write(f, "  \}\n")
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")

end



function writeLayer2File(f, L::SoftmaxWithLossLayer)
    
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"SoftmaxWithLoss\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom[j], "\"\n"))
    end
    write(f, "\}\n")
end



function writeLayer2File(f, L::HingeLossLayer)
   
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"HingeLoss\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom[j], "\"\n"))
    end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::SigmoidCrossEntropyLossLayer)
   
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"SigmoidCrossEntropyLoss\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom[j], "\"\n"))
    end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end

function writeLayer2File(f, L::InfogainLossLayer)
   
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"InfogainLoss\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom[j], "\"\n"))
    end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end

function writeLayer2File(f, L::SigmoidLayer)
  
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Sigmoid\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end
    #for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom, "\"\n"))
    #end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::TanHLayer)
  
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"TanH\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end
    #for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom, "\"\n"))
    #end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end



function writeLayer2File(f, L::AbsValLayer)
   
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"AbsVal\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end
    #for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom, "\"\n"))
    #end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end



function writeLayer2File(f, L::BNLLLayer)
  
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"BNLL\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end
    #for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom, "\"\n"))
    #end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::PowerLayer)
   
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Power\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end
    #for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom, "\"\n"))
    #end
    write(f, "power_param \{\n")
    write(f, string("  power: ", L.power, "\n"))
    write(f, string("  scale: ", L.scale, "\n"))
    write(f, string("  shift: ", L.shift, "\n"))
    write(f, "  \}\n")
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::ImageDataLayer)
  
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"ImageData\"\n")
    #for i=1:length(L.top)
        write(f, string("  top: \"", L.top, "\"\n"))
    #end

    write(f, "  data_param \{\n")
    write(f, string("    source: \"", L.source, "\"\n"))
    write(f, string("    batch_size: ", L.batch_size, "\n"))
    write(f, string("    rand_skip: ", L.rand_skip, "\n"))
    write(f, string("    shuffle: \"", L.shuffle, "\"\n"))
    
    if L.new_height > 0 && L.new_width > 0
        write(f, string("    new_height: ", L.new_height, "\n"))
        write(f, string("    new_width: ", L.new_width, "\n"))
    end
    
    write(f, "  \}\n")
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end


function writeLayer2File(f, L::ConvLayer)
       write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Convolution\"\n")
   
    write(f, string("  top: \"", L.top, "\"\n"))
    write(f, string("  bottom: \"", L.bottom, "\"\n"))
    write(f, "  param \{")
    write(f, string("  lr_mult: ", L.lr_mult_filter, " decay_mult: ", L.decay_mult_filter))
    write(f, "  \}\n")
    write(f, "  param \{")
    write(f, string("  lr_mult: ", L.lr_mult_bias, " decay_mult: ", L.decay_mult_filter))
    write(f, "  \}\n")
    write(f, "  convolution_param \{\n")
    write(f, string("    num_output: ", L.num_output, "\n"))
    if L.kernel_size > 0 && L.kernel_size_h==0 && L.kernel_size_w==0
        write(f, string("    kernel_size: ", L.kernel_size, "\n"))
    elseif L.kernel_size == 0 && L.kernel_size_h>0 && L.kernel_size_w>0
        write(f, string("    kernel_size_h: ", L.kernel_size_h, "\n"))
        write(f, string("    kernel_size_w: ", L.kernel_size_w, "\n"))
    end
    if L.stride > 0 && L.stride_h==0 && L.stride_w==0
        write(f, string("    stride: ", L.stride, "\n"))
    elseif L.stride == 0 && L.stride_h>0 && L.stride_w>0
        write(f, string("    stride_h: ", L.stride_h, "\n"))
        write(f, string("    stride_w: ", L.stride_w, "\n"))
    end
    if L.pad > 0 && L.pad_h==0 && L.pad_w==0
        write(f, string("    pad: ", L.pad, "\n"))
    elseif L.pad == 0 && L.pad_h>0 && L.pad_w>0
        write(f, string("    pad_h: ", L.pad_h, "\n"))
        write(f, string("    pad_w: ", L.pad_w, "\n"))
    end

    write(f, string("    weight_filler \{ \n"))
    write(f, string("      type: \"", L.weight_filler_type, "\"\n"))
    if L.weight_filler_type == "gaussian"
        write(f, string("      std: ", L.std_gausian_filler, "\n"))
    end
    write(f, "    \}\n")
    write(f, string("    bias_filler \{ \n"))
    write(f, string("      type: \"", L.bias_filler_type, "\"\n"))
    if L.bias_filler_type == "constant"
        write(f, string("      value: ", L.bias_filler_value, "\n")) # EVTL ALS INT64 Deklarieren!!!
    end
    write(f, "    \}\n")
    write(f, "  \}\n")
    
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    write(f, "\}\n")
end



function writeLayer2File(f, L::DataBaseLayer)
  
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Data\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end

    write(f, "  data_param \{\n")
    for j=1:length(L.source)
        write(f, string("    source: \"", L.source[j], "\"\n"))
    end
    write(f, string("    batch_size: ", L.batch_size, "\n"))
    write(f, string("    backend: ", L.backend, "\n"))
    write(f, "  \}\n")
    
    if L.scale > 0.0
        write(f, "  transform_param \{\n")
        write(f, string("    scale: ", L.scale, "\n"))
        write(f, "  \}\n")
    end
    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    

    write(f, "\}\n")
end


function writeLayer2File(f, L::AccuracyLayer)
  
    write(f, "layer \{ \n")
    write(f, string("  name: \"", L.name, "\"\n"))
    write(f, "  type: \"Accuracy\"\n")
    for i=1:length(L.top)
        write(f, string("  top: \"", L.top[i], "\"\n"))
    end
    for j=1:length(L.bottom)
        write(f, string("  bottom: \"", L.bottom[j], "\"\n"))
    end

    if ~isequal(L.phase ,"")
        write(f, "  include \{\n")
        write(f, string("    phase: ", L.phase, "\n"))
        write(f, "  \}\n")
    end
    

    write(f, "\}\n")
end






