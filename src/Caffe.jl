module Caffe


# Abstract type definition
abstract AbstractNeuralNet
abstract AbstractLayer <: AbstractNeuralNet


# Definition of the neural network type
type NeuralNet <: AbstractNeuralNet
    name::ASCIIString
    Layers::Array{AbstractLayer,1}  
end

function Net(;name="Net")
    n=NeuralNet(name, AbstractLayer[])
    return n
    
end

function NameInUse(n::NeuralNet, name::ASCIIString)
    b::Bool=false
    
    if length(n.Layers)>0
        for i=1:length(n.Layers)
            if n.Layers[i].name == name
                b=true
            end
        end
    end

    return b

end


include("Layers.jl")
include("CaffeSolver.jl")
include("FileIO.jl")


localpath=string(dirname(Base.source_path()), "/Files")




function solve(s::CaffeSolver, n::NeuralNet)
   
    currentdir=pwd();
    
    if isfile(string(localpath, "/solver.prototxt"))
        rm(string(localpath, "/solver.prototxt"))
    end

    
    
    
    if isfile(string(localpath, "/net.prototxt"))
        rm(string(localpath, "/net.prototxt"))
    end
    
    f=open(string(localpath, "/net.prototxt"), "w")

    write(f, string("name: \"", n.name, "\"\n" )) #Gesamte Netzwerkname
    for I=1:length(n.Layers)
        writeLayer2File(f, n.Layers[I])
    
    end
    close(f)


    ##### write solver prototxt #####
    path2net=localpath
    #writeSolver2File(s, path2net)
 

    if isfile(string(localpath, "/solver.prototxt"))
        rm(string(localpath, "/solver.prototxt"))
    end

    f1=open(string(localpath, "/solver.prototxt"), "w")

    write(f1, "# Automaticly generated file by the Julia Caffe interface\n")
    write(f1, string("net: \"", path2net, "/net.prototxt\"\n"))
    write(f1, "\n")
    write(f1,  string("base_lr: ", s.base_lr, "\n"))
    write(f1,  string("momentum: ", s.momentum, "\n"))
    write(f1,  string("weight_decay: ", s.weight_decay, "\n"))
    
    
    write(f1,  string("lr_policy: \"", s.lr_policy, "\"\n"))
    if s.lr_policy == "step"
        write(f1,  string("gamma: ", s.gamma, "\n"))
    elseif s.lr_policy == "inv"
        write(f1,  string("gamma: ", s.gamma, "\n"))
        write(f1,  string("power: ", s.power, "\n")) #EVTL FALSCH!!!!!
    end
    
    
    write(f1,  string("test_iter: ", s.test_iter, "\n"))
    write(f1,  string("test_interval: ", s.test_interval, "\n"))
    write(f1,  string("display: ", s.display, "\n"))
    write(f1,  string("max_iter: ", s.max_iter, "\n"))
    write(f1,  string("snapshot: ", s.snapshot, "\n"))
    write(f1,  string("snapshot_prefix: \"", s.snapshot_prefix, "\"\n"))
    write(f1,  string("solver_mode: ", s.solver_mode, "\n"))

    close(f1)



    # execute caffe
    println("Start training the network.")
    tic()
    run(`caffe train -solver=$(string(localpath, "/solver.prototxt"))`)
    b=toq();
    
    if b>60
        println("Training of the current network done succesfully. \nIt took $(round(b*100/60)/100) min. ")
    else
        println("Training of the current network done succesfully. \nIt took $(round(b*100)/100) sec. ")
    end


    
end




function solve(s::CaffeSolver, n::NeuralNet, logFile::ASCIIString)
   
    currentdir=pwd()
    
    if isfile(string(localpath, "/solver.prototxt"))
        rm(string(localpath, "/solver.prototxt"))
    end

    if isfile(string(localpath, "/net.prototxt"))
        rm(string(localpath, "/net.prototxt"))
    end
    
    f=open(string(localpath, "/net.prototxt"), "w")

    write(f, string("name: \"", n.name, "\"\n" )) #Gesamte Netzwerkname
    for I=1:length(n.Layers)
        writeLayer2File(f, n.Layers[I])
    end
    close(f)


    ##### write solver prototxt #####
    path2net=localpath

    if isfile(string(localpath, "/training.log"))
        rm(string(localpath, "/training.log"))
    end

    f0=open(string(localpath, "/training.log"), "w")
    close(f0)

    if isfile(string(localpath, "/solver.prototxt"))
        rm(string(localpath, "/solver.prototxt"))
    end

    f1=open(string(localpath, "/solver.prototxt"), "w")

    write(f1, "# Automaticly generated file by the Julia Caffe interface\n")
    write(f1, string("net: \"", path2net, "/net.prototxt\"\n"))
    write(f1, "\n")
    write(f1,  string("base_lr: ", s.base_lr, "\n"))
    write(f1,  string("momentum: ", s.momentum, "\n"))
    write(f1,  string("weight_decay: ", s.weight_decay, "\n"))
    
    
    write(f1,  string("lr_policy: \"", s.lr_policy, "\"\n"))
    if s.lr_policy == "step"
        write(f1,  string("gamma: ", s.gamma, "\n"))
    elseif s.lr_policy == "inv"
        write(f1,  string("gamma: ", s.gamma, "\n"))
        write(f1,  string("power: ", s.power, "\n")) #EVTL FALSCH!!!!!
    end
    
    
    write(f1,  string("test_iter: ", s.test_iter, "\n"))
    write(f1,  string("test_interval: ", s.test_interval, "\n"))
    write(f1,  string("display: ", s.display, "\n"))
    write(f1,  string("max_iter: ", s.max_iter, "\n"))
    write(f1,  string("snapshot: ", s.snapshot, "\n"))
    write(f1,  string("snapshot_prefix: \"", s.snapshot_prefix, "\"\n"))
    write(f1,  string("solver_mode: ", s.solver_mode, "\n"))

    close(f1)



    # execute caffe
    println("Start training the network.")

    tic()
    run(pipeline(`caffe train -solver=$(string(localpath, "/solver.prototxt"))`, stderr="$logFile"))
    b=toq();
    

    if b>60
        println("Training of the current network done succesfully. \nIt took $(round(b*100/60)/100) min. ")
    else
        println("Training of the current network done succesfully. \nIt took $(round(b*100)/100) sec. ")
    end

    println("###############################")
    println("The original log file follows.")
    println("###############################")
    println("")
    #Output the log to the STDOUT
    f2 = open(logFile)
    for ln in eachline(f2)
         print(ln)
    end
    close(f2)


    
end









export removeLayer!, Net, solve #, parseLog
export saveNet, saveSolver
export addImageDataLayer!, addDataBaseLayer!, addHDF5DataLayer!, addPowerLayer!, addBNLLLayer!
export addAbsValLayer!, addTanHLayer!, addSigmoidLayer!, addInfogainLossLayer!
export addSigmoidCrossEntropyLossLayer!, addHingeLossLayer!, addSoftmaxWithLossLayer!
export addReLULayer!, addSoftMaxLayer!, addIPLayer!, addPoolingLayer!, addConvLayer!
export addAccuracyLayer!
export CaffeAdam, CaffeSGD


end # module
