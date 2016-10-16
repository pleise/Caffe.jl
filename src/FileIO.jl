#####################################################
#### Relevant Function for File Input and Output ####
#####################################################



function saveNet(n::NeuralNet, pathname::String)
    f=open(pathname, "w")
    write(f, string("name: \"", n.name, "\"\n" )) #Gesamte Netzwerkname
    for I=1:length(n.Layers)
        writeLayer2File(f, n.Layers[I])
    end
    close(f)
end

function saveSolver(s::AbstractSolver, pathname::String, path2net::String)
    #f=open(pathname, "w")
    curPath=pwd()
    cd(pathname)


    writeSolver2File(s, path2net)
    cd(curPath)
    #close(f)
end




function writeSolver2File(s::CaffeSGD, path2net)

    if path2net==""
	error("Path to net.prototxt not definend!")
    end


    if isfile("solver.prototxt")
        rm("solver.prototxt")
    end

    f1=open("solver.prototxt", "w")

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
    
    
    #write(f1,  string("lr_policy: ", s.lr_policy, "\n"))
    #write(f1,  string("power: ", s.power, "\n"))
    write(f1,  string("test_iter: ", s.test_iter, "\n"))
    write(f1,  string("test_interval: ", s.test_interval, "\n"))
    write(f1,  string("display: ", s.display, "\n"))
    write(f1,  string("max_iter: ", s.max_iter, "\n"))
    write(f1,  string("snapshot: ", s.snapshot, "\n"))
    write(f1,  string("snapshot_prefix: \"", s.snapshot_prefix, "\"\n"))
    write(f1,  string("solver_mode: ", s.solver_mode, "\n"))

    close(f1)
    
end



#### Additional functions: readNet(path::String), readSolver(path::String)
#### Implementation follows soon!



