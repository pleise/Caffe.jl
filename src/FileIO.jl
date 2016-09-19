#####################################################
#### Relevant Function for File Input and Output ####
#####################################################



function saveNet(n::NeuralNet, pathname::ASCIIString)
    f=open(pathname, "w")
    write(f, string("name: \"", n.name, "\"\n" )) #Gesamte Netzwerkname
    for I=1:length(n.Layers)
        writeLayer2File(f, n.Layers[I])
    end
    close(f)
end

function saveSolver(s::AbstractSolver, pathname::ASCIIString, path2net::ASCIIString)
    f=open(pathname, "w")
    writesolver2file(s, path2net)
    close(f)
end




#### Additional functions: readNet(path::ASCIIString), readSolver(path::ASCIIString)
#### Implementation follows soon!



