import Pkg; Pkg.add(["DifferentialEquations", "DiffEqGPU", "CUDA", "StaticArrays"])
using DifferentialEquations, DiffEqGPU, StaticArrays, CUDA, Base.Iterators

function fEI_kuramoto_reduced_cos_asym(init_states, paras, t)

    xE, yE, xI, yI = init_states
    omega_e, omega_i, omega_f, f_str, ek, k, gamma = paras

    lagE = omega_e - omega_f + ek - k
    lagI = omega_i - omega_f + k - ek

    dxE = -gamma*xE - lagE * yE + (ek/2)*yE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*yI - 2*xE*yE*xI - yI) + f_str*yE*(xE-1)
    dyE = -gamma*yE + lagE * xE - (ek/2)*xE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*xI + 2*xE*yE*yI + xI) + f_str*xE - (f_str/2)*(1 + xE^2 - yE^2)
    dxI = -gamma*xI - lagI * yI - (ek/2)*yI*(xI^2 + yI^2 + 1) + (k/2)*(-(xI^2 - yI^2)*yE + 2*xI*yI*xE + yE)
    dyI = -gamma*yI + lagI * xI + (ek/2)*xI*(xI^2 + yI^2 + 1) - (k/2)*((xI^2 - yI^2)*xE + 2*xI*yI*yE + xE)

    return SVector{4, Float32}(dxE, dyE, dxI, dyI)
end

function fEI_kuramoto_reduced_cos_sym(init_states, paras, t)

    xE, yE, xI, yI = init_states
    omega_e, omega_i, omega_f, f_str, ek, k, gamma = paras

    lagE = omega_e - omega_f + ek - k
    lagI = omega_i - omega_f + k - ek

    dxE = -gamma*xE - lagE * yE + (ek/2)*yE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*yI - 2*xE*yE*xI - yI) + f_str*yE*(xE-1)
    dyE = -gamma*yE + lagE * xE - (ek/2)*xE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*xI + 2*xE*yE*yI + xI) + f_str*xE - (f_str/2)*(1 + xE^2 - yE^2)
    dxI = -gamma*xI - lagI * yI - (ek/2)*yI*(xI^2 + yI^2 + 1) + (k/2)*(-(xI^2 - yI^2)*yE + 2*xI*yI*xE + yE) + f_str*yI*(xI-1)
    dyI = -gamma*yI + lagI * xI + (ek/2)*xI*(xI^2 + yI^2 + 1) - (k/2)*((xI^2 - yI^2)*xE + 2*xI*yI*yE + xE) + f_str*xI - (f_str/2)*(1 + xI^2 - yI^2)

    return SVector{4, Float32}(dxE, dyE, dxI, dyI)
end

function fEI_kuramoto_reduced_sin_sym(init_states, paras, t)

    xE, yE, xI, yI = init_states
    omega_e, omega_i, omega_f, f_str, ek, k, gamma = paras

    lagE = omega_e - omega_f + ek - k
    lagI = omega_i - omega_f + k - ek

    dxE = -gamma*xE - lagE*yE + (ek/2)*yE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*yI - 2*xE*yE*xI - yI) + (f_str/2)*(1 - xE^2 + yE^2)
    dyE = -gamma*yE + lagE*xE - (ek/2)*xE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*xI + 2*xE*yE*yI + xI) - f_str*xE*yE
    dxI = -gamma*xI - lagI*yI - (ek/2)*yI*(xI^2 + yI^2 + 1) + (k/2)*(-(xI^2 - yI^2)*yE + 2*xI*yI*xE + yE) + (f_str/2)*(1 - xI^2 + yI^2)
    dyI = -gamma*yI + lagI*xI + (ek/2)*xI*(xI^2 + yI^2 + 1) - (k/2)*((xI^2 - yI^2)*xE + 2*xI*yI*yE + xE) - f_str*xI*yI

    return SVector{4, Float32}(dxE, dyE, dxI, dyI)
end

function fEI_kuramoto_reduced_sin_asym(init_states, paras, t)

    xE, yE, xI, yI = init_states
    omega_e, omega_i, omega_f, f_str, ek, k, gamma = paras

    lagE = omega_e - omega_f + ek - k
    lagI = omega_i - omega_f + k - ek

    dxE = -gamma*xE - lagE*yE + (ek/2)*yE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*yI - 2*xE*yE*xI - yI) + (f_str/2)*(1 - xE^2 + yE^2)
    dyE = -gamma*yE + lagE*xE - (ek/2)*xE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*xI + 2*xE*yE*yI + xI) - f_str*xE*yE
    dxI = -gamma*xI - lagI*yI - (ek/2)*yI*(xI^2 + yI^2 + 1) + (k/2)*(-(xI^2 - yI^2)*yE + 2*xI*yI*xE + yE)
    dyI = -gamma*yI + lagI*xI + (ek/2)*xI*(xI^2 + yI^2 + 1) - (k/2)*((xI^2 - yI^2)*xE + 2*xI*yI*yE + xE)

    return SVector{4, Float32}(dxE, dyE, dxI, dyI)
end

# group_type: 1 = E only | 2 = I only | 3 = both
function par_sweep(x, y, times, paras, model_type, group_type = 1)
    xjl = Float32.(x) # is it necessary?
    yjl = Float32.(y)

    times2save = Float32.(times)
    steps = Int64(length(times2save))
    tot_t = maximum(times2save)
    settle_t = Int64(steps/2)

    if model_type == 0
        fun = fEI_kuramoto_reduced_cos_asym
    elseif model_type == 1
        fun = fEI_kuramoto_reduced_cos_sym
    elseif model_type == 2
        fun = fEI_kuramoto_reduced_sin_sym
    elseif model_type == 3
        fun = fEI_kuramoto_reduced_sin_asym
    else error("Passed the wrong model type, select an existing one")
    end
    println("using model type ", string(fun))

    if !(group_type in (1, 2, 3))
        error("Passed the wrong group type, select 1 (E only), 2 (I only) or 3 (both)")
    end
    save_E = group_type == 1 || group_type == 3
    save_I = group_type == 2 || group_type == 3
    println("saving group(s): ", (save_E && save_I) ? "E and I" : (save_E ? "E" : "I"))

    # only the requested states are pulled off the gpu
    state_idxs = Int64[]
    save_E && append!(state_idxs, [1, 2])
    save_I && append!(state_idxs, [3, 4])
    # row of xE / xI inside the returned batch array (y is the following row)
    rowE = 1
    rowI = save_E ? 3 : 1

    omega_e, omega_i, omega_f, f_str, ek, k_init, k_stim, gamma = paras

    time_for_stable = SVector{2, Float32}(0.0f0, 1532.3f0)
    time = SVector{2, Float32}(0.0f0, tot_t)
    init = SVector{4, Float32}(0.1f0, 0.1f0, 0.1f0, 0.1f0)
    # ugly but is needed to preallocate on gpu efficiently
    paras_init = SVector{7, Float32}(omega_e, omega_i, 0.0f0, 0.0f0, ek, k_init, gamma)
    prob_init = ODEProblem(fun, init, time_for_stable, paras_init)
    sol_init = solve(prob_init, Tsit5())
    stable_init = sol_init.u[end]
    println("last 10 secs: ", sol_init.u[end-10:end])
    println(stable_init)
    println("order: ", sqrt(stable_init[1]^2 + stable_init[2]^2))
    println("order I: ", sqrt(stable_init[3]^2 + stable_init[4]^2))

    paras_base = SVector{7, Float32}(omega_e, omega_i, 0.0f0, 0.0f0, ek, k_stim, gamma)
    prob_base = ODEProblem(fun, stable_init, time, paras_base)

    x_len = Int64(length(xjl))
    y_len = Int64(length(yjl))
    total_sols = Int64(x_len*y_len)
    batchsize = 100000
    # unused groups get zero sized arrays, so nothing is allocated for them
    alloc_re(flag) = flag ? Array{Float32}(undef, steps, x_len, y_len) : Array{Float32}(undef, 0, 0, 0)
    alloc_comp(flag) = flag ? Array{Float32}(undef, 2, settle_t, x_len, y_len) : Array{Float32}(undef, 0, 0, 0, 0)
    final_reE = alloc_re(save_E)
    final_compE = alloc_comp(save_E)
    final_reI = alloc_re(save_I)
    final_compI = alloc_comp(save_I)

    # this is not good for speed, as it executes the solutions by batches, creating unnecessary transfers between cpu-gpu
    # and the problem function/grid has to be created and sent ot gpu for every batch
    # but it is needed to save sys memory: only the final array [(xE, yE), timesteps, len(x), len(y)] is created
    # and populated as the batches are completed, every batch result is then cleared after each iteration
    println("starting $(total_sols) sols")
    for vec_i in 1:batchsize:total_sols

        start_i = vec_i
        end_i = Int64(min(vec_i + batchsize - 1, total_sols))
        curr_batchsize = end_i - start_i + 1

        # has to be created for every batch, since it has to have access the batch xjl, yjl by the batch index
        function prob_fun(prob_base, ctx)
            idx = ctx.sim_id
            g_i = start_i + idx - 1
            xidx = mod1(g_i, x_len)
            yidx = cld(g_i, x_len)

            omega_f_curr = xjl[xidx]
            f_str_curr = yjl[yidx]

            paras_curr = SVector{7, Float32}(prob_base.p[1], prob_base.p[2], omega_f_curr, f_str_curr, 
                prob_base.p[5], prob_base.p[6], prob_base.p[7])

            return remake(prob_base, p = paras_curr)
        end
        # same
        prob_grid = EnsembleProblem(prob_base, prob_func = prob_fun)

        @time sol_raw_batch = solve(prob_grid, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
            trajectories = curr_batchsize, saveat = times2save, save_idxs = state_idxs)

        struct_array = Array(sol_raw_batch)

        Threads.@threads for loc_i in 1:curr_batchsize
            glob_i = start_i + loc_i - 1

            xi = mod1(glob_i, x_len)
            yi = cld(glob_i, x_len)

            @inbounds for t in 1:steps

                if save_E
                    xE_curr = struct_array[rowE, t, loc_i]
                    yE_curr = struct_array[rowE+1, t, loc_i]
                    curr_reE = sqrt(xE_curr^2 + yE_curr^2)
                    final_reE[t, xi, yi] = curr_reE

                    if t > settle_t
                        tidx = t - settle_t
                        final_compE[1, tidx, xi, yi] = xE_curr
                        final_compE[2, tidx, xi, yi] = yE_curr
                    end
                end

                if save_I
                    xI_curr = struct_array[rowI, t, loc_i]
                    yI_curr = struct_array[rowI+1, t, loc_i]
                    curr_reI = sqrt(xI_curr^2 + yI_curr^2)
                    final_reI[t, xi, yi] = curr_reI

                    if t > settle_t
                        tidx = t - settle_t
                        final_compI[1, tidx, xi, yi] = xI_curr
                        final_compI[2, tidx, xi, yi] = yI_curr
                    end
                end
            end
        end

        struct_array = nothing
        sol_raw_batch = nothing
        GC.gc()
        CUDA.reclaim()

    end

    if group_type == 1
        return final_reE, final_compE
    elseif group_type == 2
        return final_reI, final_compI
    else
        return final_reE, final_compE, final_reI, final_compI
    end
end