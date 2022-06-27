function diis(;
              max_depth=10,
              adaptative_depth=false,
              #threshold?
              )

    function instantiate(infos)
        x_list = [infos.ζ.Φ]
        s_list = []
        depth = 0
        (;depth, x_list, s_list)
    end

    function interpolate(x, residual, diis_infos)
        # Choose depth
        depth = diis_infos.depth
        (depth==0) && (return x)

        # Extract previous x and s
        x_list = diis_info.x_list[end-depth+1:end]
        s_list = diis_info.s_list[end-depth+1:end]

        # Construct and solve least square system
        A = hcat([dot(s1, s2) for s1 in s_list, s2 in s_list]...)
        B = 2 .* dot.(Ref(residual), s_list)
        α_opti = (1//2)(A\B) #(1//2)*cg(A,B,abstol = 1e-10)

        # Assemble new guess    
        distances = [x[i+1] .- x[i] for i in 1:(length(x_list)-1)]
        x_new = x .- sum(α_opti[end-i] .* dx[i] for i in 1:length(distances))
    end
    interpolate(info) = interpolate(info.residual, info.diis_info)

    function update_infos(x, diis_info)
        nothing
        # depth = diis_infos.depth + one(Int64)
        # r_new = x .- diis_info.x_list[end]
        # s_new = r_new - diis_info.
        # (; depth, 
    end

    (;instantiate, interpolate, update_infos)
end
