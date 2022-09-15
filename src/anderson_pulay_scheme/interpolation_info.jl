# TO BE CONTINUED

# function residual_interpolation(info; max_depth=10)
#     function init(info)
#         depth = zero(Int64)
#         r_list = info.∇E_prev_norm
#         s_list = Float64[]
#         x_list = []
#         (;depth, max_depth, r_list, s_list, x_list)
#     end

#     function choose_depth(itp_info)
#         (itp_info.depth=max_depth) && (return itp_info)
#         merge(itp_info, (;depth=itp_info.depth+one(Int64)))
#     end

#     function itp_guess(ζ::ROHFState, x, residual, info, itp_info)
#         # Handle zero depth case
#         depth = itp_info.depth
#         x_list = project_tangent.(Ref(ζ.M.mo_numbers), Ref(ζ.Φ),
#                                   itp_info.x_list[end-depth+1:end])
#         push!(x_list, x)
#         s_list = itp_info.s_list[end-depth+1:end]
#         r_list = itp_info.r_list[end-depth+1:end]

#         (depth==0) && (return zero(Int64))
        
#         # Solve least square residual interpolation system
#         A = [s1*s2 for s1 in s_list, s2 in s_list]
#         B = residual .* s_list
#         C = A\B
        
#         # Assemble guess
#         x_itp = x .- sum(C[i]*(x_list[i]-x_list[i-1]) for i in 1:depth+1)

#         # update itp_info
#         x_itp, merge(itp_info; x_list=x_list)
#     end

#     # function update_itp_info(itp_info)
        
#     # end
# end
