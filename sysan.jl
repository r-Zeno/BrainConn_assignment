using BifurcationKit, Setfield, LinearAlgebra, ForwardDiff, DifferentialEquations

function fEI_cos(u, p)
    xE, yE, xI, yI = u
    ek, k, omega_e, omega_i, gamma, f_str, omega_f = p.ek, p.k, p.omega_e, p.omega_i, p.gamma, p.f_srt, p.omega_f

    lagE = omega_e - omega_f + ek - k
    lagI = omega_i - omega_f + k - ek

    dxE = -gamma*xE - lagE * yE + (ek/2)*yE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*yI - 2*xE*yE*xI - yI) + f_str*yE*(xE-1)
    dyE = -gamma*yE + lagE * xE - (ek/2)*xE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*xI + 2*xE*yE*yI + xI) + f_str*xE - (f_str/2)*(1 + xE^2 - yE^2)
    dxI = -gamma*xI - lagI * yI - (ek/2)*yI*(xI^2 + yI^2 + 1) + (k/2)*(-(xI^2 - yI^2)*yE + 2*xI*yI*xE + yE) + f_str*yI*(xI-1)
    dyI = -gamma*yI + lagI * xI + (ek/2)*xI*(xI^2 + yI^2 + 1) - (k/2)*((xI^2 - yI^2)*xE + 2*xI*yI*yE + xE) + f_str*xI - (f_str/2)*(1 + xI^2 - yI^2)
    return [dxE, dyE, dxI, dyI]
end

function fEI_sin(u, p)
    xE, yE, xI, yI = u
    ek, k, omega_e, omega_i, gamma, f_str, omega_f = p.ek, p.k, p.omega_e, p.omega_i, p.gamma, p.f_srt, p.omega_f

    lagE = omega_e - omega_f + ek - k
    lagI = omega_i - omega_f + k - ek

    dxE = -gamma*xE - lagE*yE + (ek/2)*yE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*yI - 2*xE*yE*xI - yI) + (f_str/2)*(1 - xE^2 + yE^2)
    dyE = -gamma*yE + lagE*xE - (ek/2)*xE*(xE^2 + yE^2 + 1) + (k/2)*((xE^2 - yE^2)*xI + 2*xE*yE*yI + xI) - f_str*xE*yE
    dxI = -gamma*xI - lagI*yI - (ek/2)*yI*(xI^2 + yI^2 + 1) + (k/2)*(-(xI^2 - yI^2)*yE + 2*xI*yI*xE + yE)
    dyI = -gamma*yI + lagI*xI + (ek/2)*xI*(xI^2 + yI^2 + 1) - (k/2)*((xI^2 - yI^2)*xE + 2*xI*yI*yE + xE)
    return [dxE, dyE, dxI, dyI]
end

function get_entrainment_boundaries(model_choice::String)
    f_model = model_choice == "cos" ? fEI_cos : fEI_sin
    ode_f(u, p, t) = f_model(u, p)

    f_srt_slices = [0.20, 1.5] 
    tasks_2d = [] 

    opts_1d = ContinuationPar(
        dsmax = 0.01, dsmin = 1e-4, ds = 0.01,
        max_steps = 5000, p_min = 1.0, p_max = 5.0, 
        detect_bifurcation = 3
    )

    for f_val in f_srt_slices
        init_paras = (ek = 0.0, k = 0.5, omega_e = 1.5, omega_i = 0.5, gamma = 0.1, f_srt = f_val, omega_f = 1.15)
        u0 = [0.3, 0.0, 0.1, 0.0]

        stable_prob = ODEProblem(ode_f, u0, (0.0, 3000.0), init_paras)
        sol = solve(stable_prob, Tsit5())
        u0_stable = sol.u[end] 
        
        lens_omf = @optic _.omega_f
        prob = BifurcationProblem(f_model, u0_stable, init_paras, lens_omf)

        try
            br_1d = continuation(prob, PALC(), opts_1d, verbosity = 0; bothside = true)
            valid_bifs = findall(pt -> pt.type in (:fold, :hopf), br_1d.specialpoint)
            
            for idx in valid_bifs
                push!(tasks_2d, (br_1d, idx, f_val))
            end
        catch e
        end
    end

    lens2 = @optic _.f_srt
    opts_2d = ContinuationPar(
        dsmax = 0.005, ds = 0.001, dsmin = 1e-5, 
        max_steps = 5000, p_min = 0.0, p_max = 5.0,
        detect_bifurcation = 2, newton_options = NewtonPar(tol = 1e-9)
    )

    boundaries = []
    
    for (i, (br_1d, idx, f_val)) in enumerate(tasks_2d)
        br_2d = continuation(br_1d, idx, lens2, opts_2d; bothside = true)
        if length(br_2d.branch.omega_f) > 1
            push!(boundaries, hcat(br_2d.branch.omega_f, br_2d.branch.f_srt))
        end
    end
    
    return boundaries
end