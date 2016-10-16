# Abstract solver types
abstract AbstractSolver
abstract CaffeSolver <: AbstractSolver

type CaffeSGD <: CaffeSolver
    base_lr::Float64# = 0.01
    
    momentum::Float64# = 0.9
    weight_decay::Float64# = 0.0005
    
    # The learning rate policy
    lr_policy::String# = "\"inv\"" or "\"fixed\""
    gamma::Float64# = 0.0001
    power::Float64# = 0.75
    
    # Test 
    test_iter::Int64# = 100
    test_interval::Int64# = 500
    
    # Display iterations
    display::Int64# = 100
    
    # The maximum number of iterations
    max_iter::Int64# = 10000
    
    # snapshot intermediate results
    snapshot::Int64# = 5000
    snapshot_prefix::String# = /example/mnist
    
    # solver mode: CPU or GPU
    solver_mode::String# = "CPU"
    #solver_type::String# ="\"SGD\"" or Adam, AdamGrad, Nestreov, RMSProp
    
 
end


type CaffeAdam <: CaffeSolver
    base_lr::Float64# = 0.01
    
    momentum::Float64# = 0.9
    momentum2::Float64# = 0.9999
    
    #weight_decay::Float64# = 0.0005
    
    # The learning rate policy
    lr_policy::String# = "\"inv\"" or "\"fixed\""
    #gamma::Float64# = 0.0001
    #power::Float64# = 0.75
    
    # Test 
    test_iter::Int64# = 100
    test_interval::Int64# = 500 # EVTL AUCH RAUS!!!
    
    # Display iterations
    display::Int64# = 100
    
    # The maximum number of iterations
    max_iter::Int64# = 10000
    
    # snapshot intermediate results
    snapshot::Int64# = 5000
    snapshot_prefix::String# = /example/mnist
    
    # solver mode: CPU or GPU
    solver_mode::String# = "CPU"
    #solver_type::String# ="\"SGD\"" or Adam, AdamGrad, Nestreov, RMSProp
    
end


function CaffeAdam(;base_lr=0.0, momentum=0.0, momentum2=0.0, lr_policy="", 
 test_iter=0, test_interval=0, display=0, max_iter=0, 
    snapshot=0, snapshot_prefix=pwd(), solver_mode="CPU")
    
    s=CaffeAdam(base_lr, momentum, momentum2, lr_policy, 
  test_iter, test_interval, display, max_iter, 
    snapshot, snapshot_prefix, solver_mode)
    
    return s
end

function CaffeSGD(;base_lr=0.0, momentum=0.0, weight_decay=0.0, lr_policy="", 
gamma=0.0,  power=0.0, test_iter=0, test_interval=0, display=0, max_iter=0, 
    snapshot=0, snapshot_prefix=pwd(), solver_mode="CPU")
    
    s=CaffeSGD(base_lr, momentum, weight_decay, lr_policy, 
    gamma, power, test_iter, test_interval, display, max_iter, 
    snapshot, snapshot_prefix, solver_mode)
    
    return s
end



