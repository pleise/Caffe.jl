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




"""    parseLog(LogFile::ASCIIString) 
Extracts relevant values of  the training and testing process. It returns two matrices. 
The first shows the test data and the  second the training data.

Usage:

testOut, trainOut = parseLog(LogFile::ASCIIString) 

The columns of each matrix have the following meaning:

trainOut = [(Iteration) (Time in seconds) (Learning rate) (Loss)]

testOut= [(Iteration) (Time in seconds) (Accuracy) (Loss)]
 """
function parseLog(log::ASCIIString)
    
    f=open(log)
    l=readlines(f)
    close(f)
    
    
    ##################################
    #####  Generate train output  ####
    ##################################
    k=0
    for i=1:length(l)
        if ismatch(r"Solving", l[i])
            # get the solving keyword to know that the solving started and get the start time
            k=i

        end
    end

    # Extract start time

    if ismatch(r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+.\d+)", l[k])
        a=match(r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+.\d\d\d)", l[k])
    end


    startTime=DateTime(a.match, "H:M:S.s")

    
    iteration=[]
    loss=[]
    time_loss=[]

    for i = 1:length(l)
        if ismatch(r"Iteration (\d+), loss = ([\.\deE+-]+)", l[i])
            It=match(r"Iteration (\d+), loss = ([\.\deE+-]+)", l[i]) #get Iteration and loss
            #println(It)
            # now get the 
            push!(iteration, parse(Int64, It[:1]))
            push!(loss, parse(Float64, It[:2]))
            #break

            # get time

            b=match(r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+.\d\d\d)", l[i])
            push!(time_loss, DateTime(b.match, "H:M:S.s"))




        end

    end
    
    lr=[]
    iteration_lr=[]
    for i = 1:length(l)
        if ismatch(r"Iteration (\d+), lr = ([\.\deE+-]+)", l[i])
            It=match(r"Iteration (\d+), lr = ([\.\deE+-]+)", l[i]) #get Iteration and loss
            #println(It)
            # now get the 
            push!(iteration_lr, parse(Int64, It[:1]))
            push!(lr, parse(Float64, It[:2]))
            #break
        end

    end
    
    # Time calculation
    time_milliseconds=[]

    for i=1:length(time_loss)
        a=time_loss[i]-startTime
        push!(time_milliseconds, (a.value))

    end
    time_seconds=time_milliseconds/1000
    
    # generate the train output

    trainOut=zeros(length(iteration), 4)
    for i=1:4
        for j=1:length(iteration)
            trainOut[j,i]=NaN
        end
    end
    # num Iterations, seconds, LR, loss

    trainOut[:,1]=iteration;
    trainOut[:,4]=loss;
    trainOut[:,2]=time_seconds;

    for i=1:size(trainOut,1)
        for j=1:length(iteration_lr) 
            if iteration_lr[j]==trainOut[i,1]
                trainOut[i,3]=lr[j]
            end

        end

    end
    
    
    ##################################
    #####   Generate test output  ####
    ##################################
    
    iteration_test=[]
    loss_test=[]
    accuracy_test=[]
    time_test=[]

    for i = 1:length(l)
        if ismatch(r"Iteration (\d+), Testing net", l[i])

            b=match(r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+.\d\d\d)", l[i])
            push!(time_test, DateTime(b.match, "H:M:S.s"))


            It=match(r"Iteration (\d+), Testing net", l[i]) #get Iteration and loss
            #println(It)

            push!(iteration_test, parse(Int64, It[:1]))


            # get accuracy and loss and time
            for j=i+1:i+5
                if ismatch(r"Test net output #(\d+): loss = ([\.\deE+-]+)", l[j])
                    M=match(r"Test net output #(\d+): loss = ([\.\deE+-]+)", l[j])
                    push!(loss_test, parse(Float64, M[:2]))
                    #println(M)
                #else
                    #push!(loss_test, NaN)
                end
            end

            for j=i+1:i+5
                if ismatch(r"Test net output #(\d+): accuracy = ([\.\deE+-]+)", l[j])
                    M=match(r"Test net output #(\d+): accuracy = ([\.\deE+-]+)", l[j])
                    push!(accuracy_test, parse(Float64, M[:2]))
                    #println(M)
                #else
                    #push!(loss_test, NaN)
                end
            end


        end

    end

    time_milliseconds_test=[]

    for i=1:length(time_test)
        a=time_test[i]-startTime
        push!(time_milliseconds_test, (a.value))

    end
    time_seconds_test=time_milliseconds_test/1000

    testOut=zeros(length(iteration_test), 4)
    for i=1:4
        for j=1:length(iteration_test)
            testOut[j,i]=NaN
        end
    end

    testOut[:,1]=iteration_test
    testOut[:,2]=time_seconds_test
    testOut[:,3]=accuracy_test
    testOut[:,4]=loss_test
    
    
    
    return testOut, trainOut
    
    
end





export removeLayer!, Net, solve, parseLog
export saveNet, saveSolver
export addImageDataLayer!, addDataBaseLayer!, addHDF5DataLayer!, addPowerLayer!, addBNLLLayer!
export addAbsValLayer!, addTanHLayer!, addSigmoidLayer!, addInfogainLossLayer!
export addSigmoidCrossEntropyLossLayer!, addHingeLossLayer!, addSoftmaxWithLossLayer!
export addReLULayer!, addSoftMaxLayer!, addIPLayer!, addPoolingLayer!, addConvLayer!
export addAccuracyLayer!
export CaffeAdam, CaffeSGD


end # module
