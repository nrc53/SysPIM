import numpy as np
import math
import sys

# crossbar implementation for various layers
class in_cache_arch:
    def __init__(self, in_cache_config, comp_precsn, comp_mode):
        self.in_cache_config = in_cache_config
        self.comp_precsn = comp_precsn
        self.comp_mode = comp_mode
        self.comp_tile_size = in_cache_config['comp_tile_size']
        # 0.625 MB is the standard bank size, each PE is 8KB arranged in a 10x8 systolic config inside a bank
        self.num_PEs = (in_cache_config['cache_size'] / 0.625) * 10 * 8 
        # In a systolic fashio, all the PEs in 10x8 configuration finishes their local [8x8, 8x8] mat mult in 185 cycles
        self.comp_cycles = 185 

    def conv(self, input_dims, layer_params):
        kernel_dims = layer_params['kernel']
        padding = layer_params['padding']  
        stride = layer_params['stride']
        output_dim_r = round((input_dims[0] - kernel_dims[0] + padding) / stride) + 1
        output_dim_c = round((input_dims[1] - kernel_dims[1] + padding) / stride) + 1
        # Convolution is performed as block matrix multiplication by downing the input matrix 
        input_mat_cols = kernel_dims[0] * kernel_dims[1] * kernel_dims[2]
        input_mat_rows = output_dim_r * output_dim_c
        weight_mat_rows = input_mat_cols
        weight_mat_cols = kernel_dims[3]
        weight_tiles_r = math.ceil(weight_mat_cols / 8)
        weight_tiles_c = math.ceil(input_mat_cols / 8)
        num_tiled_matmuls = math.ceil(input_mat_rows / 8) * weight_tiles_c * weight_tiles_r
        if self.comp_mode == 'max_throughput':
            if num_tiled_matmuls <= self.num_PEs:
                tot_cycles = self.comp_cycles
            else: 
                tot_cycles = self.comp_cycles * math.ceil(num_tiled_matmuls/self.num_PEs)
        elif self.comp_mode == "area_efficient":
            tiles_per_subarray = (7 * 1024) / (8*8) 
            num_subarrays = math.ceil(num_tiled_matmuls / tiles_per_subarray)
            if num_subarrays <= self.num_PEs:
                tot_cycles = tiles_per_subarray * self.comp_cycles
            else:
                tot_cycles = tiles_per_subarray * self.comp_cycles * math.ceil(num_subarrays/self.num_PEs)

        num_macs_with_common_inp = kernel_dims[-1]
        output_dims = [output_dim_r, output_dim_c, num_macs_with_common_inp]

        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)
        if layer_params['activation'] == 'ReLu':
            exec_stats = {'mm_tiles':num_tiled_matmuls, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
                'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, "activation":True}
        else:
            exec_stats = {'mm_tiles':num_tiled_matmuls, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
                'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, "activation":False}
        return exec_stats

    def conv_depthwise(self, input_dims, layer_params):
        kernel_dims = layer_params['kernel']
        padding = layer_params['padding']  
        stride = layer_params['stride']
        output_dim_r = round((input_dims[0] - kernel_dims[0] + padding) / stride) + 1
        output_dim_c = round((input_dims[1] - kernel_dims[1] + padding) / stride) + 1
        # Convolution is performed as block matrix multiplication by downing the input matrix 
        input_mat_cols = kernel_dims[0] * kernel_dims[1]
        input_mat_rows = output_dim_r * output_dim_c
        weight_mat_rows = input_mat_cols
        weight_mat_cols = 1
        weight_tiles_r = math.ceil(weight_mat_cols / 8)
        weight_tiles_c = math.ceil(input_mat_cols / 8)
        num_tiled_matmuls = math.ceil(input_mat_rows / 8) * weight_tiles_c * weight_tiles_r * kernel_dims[2]
        if self.comp_mode == 'max_throughput':
            if num_tiled_matmuls <= self.num_PEs:
                tot_cycles = self.comp_cycles
            else: 
                tot_cycles = self.comp_cycles * math.ceil(num_tiled_matmuls/self.num_PEs)
        elif self.comp_mode == "area_efficient":
            tiles_per_subarray = (7 * 1024) / (8*8) 
            num_subarrays = math.ceil(num_tiled_matmuls / tiles_per_subarray)
            if num_subarrays <= self.num_PEs:
                tot_cycles = tiles_per_subarray * self.comp_cycles
            else:
                tot_cycles = tiles_per_subarray * self.comp_cycles * math.ceil(num_subarrays/self.num_PEs)

        num_macs_with_common_inp = kernel_dims[-1]
        output_dims = [output_dim_r, output_dim_c, num_macs_with_common_inp]

        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)
        
        if layer_params['activation'] == 'ReLu':
            exec_stats = {'mm_tiles':num_tiled_matmuls, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
                'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, "activation":True}
        else:
            exec_stats = {'mm_tiles':num_tiled_matmuls, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
                'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, "activation":False}
        return exec_stats        

    def fc(self, input_dims, layer_params):
        input_dims = [1, np.prod(input_dims)]
        kernel_dims = layer_params['kernel']
        output_dim_r = input_dims[0]
        output_dim_c = kernel_dims[1]
        input_mat_cols = input_dims[1]
        input_mat_rows = input_dims[0]
        weight_mat_rows = kernel_dims[0]
        weight_mat_cols = kernel_dims[1]
        weight_tiles_r = math.ceil(weight_mat_cols / 8)
        weight_tiles_c = math.ceil(input_mat_cols / 8)
        num_tiled_matmuls = math.ceil(input_mat_rows / 8) * weight_tiles_c * weight_tiles_r 
        if self.comp_mode == 'max_throughput':
            if num_tiled_matmuls <= self.num_PEs:
                tot_cycles = self.comp_cycles
            else: 
                tot_cycles = self.comp_cycles * math.ceil(num_tiled_matmuls/self.num_PEs)
        elif self.comp_mode == "area_efficient":
            tiles_per_subarray = (7 * 1024) / (8*8) 
            num_subarrays = math.ceil(num_tiled_matmuls / tiles_per_subarray)
            if num_subarrays <= self.num_PEs:
                tot_cycles = tiles_per_subarray * self.comp_cycles
            else:
                tot_cycles = tiles_per_subarray * self.comp_cycles * math.ceil(num_subarrays/self.num_PEs)

        output_dims = [input_dims[0], kernel_dims[1]]

        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)
        
        if layer_params['activation'] == 'ReLu':
            exec_stats = {'mm_tiles':num_tiled_matmuls, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
                'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, "activation":True}
        else:
            exec_stats = {'mm_tiles':num_tiled_matmuls, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
                'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, "activation":False}
        return exec_stats        

    def relu(self, input_dims):
        tot_num_ops = np.prod(input_dims) #input_dims[0] * input_dims[1] * input_dims[2]
        num_alus = 8 * self.num_PEs
        tot_cycles = math.ceil(tot_num_ops / num_alus)
        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(input_dims)
        return tot_cycles
        # exec_stats = {'exec_cycles':tot_cycles, 'output_dims':input_dims, 'input_bytes_read':inp_bytes_read, \
        #     'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}

        # return exec_stats

    def residual_add(self, input_dims):
        num_alus = 8 * self.num_PEs
        tot_num_ops = input_dims[0] * input_dims[1] * input_dims[2]
        tot_cycles = math.ceil(tot_num_ops / num_alus)
        inp_bytes_read = np.prod(input_dims) * 2
        output_bytes_written = np.prod(input_dims)
        exec_stats = {'mm_tiles':math.ceil(tot_num_ops / 8), 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':input_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def pool(self, input_dims, layer_params):
        num_alus = 8 * self.num_PEs
        kernel_dims = layer_params['kernel']
        elems_per_kernel = kernel_dims[0] * kernel_dims[1]
        ops_per_kernel = math.floor(elems_per_kernel * np.log(elems_per_kernel))
        num_kernels_r = round((input_dims[0] - kernel_dims[0]) / layer_params['stride'][0]) + 1
        num_kernels_c = round((input_dims[1] - kernel_dims[1]) / layer_params['stride'][1]) + 1
        num_kernels = num_kernels_r * num_kernels_c * input_dims[2] 
        if layer_params['pool_op'] == 'avg':            
            tot_cycles = math.ceil((ops_per_kernel * num_kernels) / num_alus)
            tot_div_ops = num_kernels
        if layer_params['pool_op'] == 'max':
            tot_cycles = math.ceil((ops_per_kernel * num_kernels) / num_alus)
        output_dims = [num_kernels_r, num_kernels_c, input_dims[2]]
        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        exec_stats = {'mm_tiles':math.ceil((ops_per_kernel * num_kernels) / 8), 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def concat(self, input_dims):
        concat_dim = 0
        for dim in input_dims:
            concat_dim += dim[-1]
        output_dims = [input_dims[0][0], input_dims[0][1], concat_dim]
        exec_stats = {'mm_tiles':0, 'exec_cycles':0, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':0, \
            'output_bytes_written':0, 'kernel_bytes_read':0}
        return exec_stats

    def elem_wise_mult(self, input_dims):
        tot_mult_ops = input_dims[0] + input_dims[1] # Specific to the LSTM implementation
        num_alus = self.num_PEs
        # 2 cycles for each mult operation
        tot_cycles = math.ceil(tot_mult_ops / num_alus) * 2
        input_bytes_read = tot_mult_ops * 2
        output_bytes_written = input_dims[0]
        exec_stats = {'mm_tiles':tot_mult_ops, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':[input_dims[0]], 'input_bytes_read':input_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def elem_wise_add(self, input_dims):
        tot_add_ops = input_dims[0]  # Specific to the LSTM implementation
        num_alus = 8 * self.num_PEs
        input_bytes_read = tot_add_ops * 2
        tot_cycles = math.ceil(tot_add_ops / num_alus)
        output_bytes_written = input_dims[0] 
        exec_stats = {'mm_tiles':math.ceil(tot_add_ops / 8), 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':[input_dims[0]], 'input_bytes_read':input_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

def in_cache_analyze_exec_stats_parallel_branches(exec_stats_list, network_config, in_cache_config, in_cache_arch):
    i = 0
    tot_exec_cycles = 0
    tot_dram_bytes_read = 0
    tot_dram_bytes_written = 0
    tot_weight_bytes = 0
    prev_layer_merged = False
    layer_merged_with_num = -1
    next_layer_read_inputs = True
    while (i < len(exec_stats_list)):
        exec_stat = exec_stats_list[i]
        overlap = False
        required_PEs = exec_stat['mm_tiles']
        #print(i, exec_stat, prev_layer_merged)
        # Check if first node/layer in the parallel (second) branch can be merged 
        if exec_stat['inp_layer'] != "prev_layer" and exec_stat['type'] != "concat":
            first_branch_layer = exec_stat['parallel_layer']
            first_branch_layer_num = network_config['layers'][first_branch_layer]['num']
            j = first_branch_layer_num
            while (j < i and exec_stats_list[j]['merged'] != True):
                layer_PEs = exec_stats_list[j]['mm_tiles']
                if required_PEs + layer_PEs <=  in_cache_arch.num_PEs:
                    overlap = True
                    exec_stats_list[j]['mm_tiles'] = required_PEs + layer_PEs
                    # Calculate the number of weight elements in merged layer [i] and add to main layer [j]
                    merged_layer_params = network_config['layers'][exec_stat['name']]
                    num_weight_elems = calc_weight_elems(merged_layer_params)
                    if "extra_weights" in exec_stats_list[j]:
                        exec_stats_list[j]['extra_weights'] += num_weight_elems
                    else:
                        exec_stats_list[j]['extra_weights'] = num_weight_elems
                    exec_stats_list[i]['merged'] = True
                    prev_layer_merged = True
                    layer_merged_with_num = j
                    #print("Merged layers "+str(i)+" and "+str(j))
                    break
                else:
                    j += 1
            if (overlap == False):
                tot_exec_cycles += exec_stat['exec_cycles']
                prev_layer_merged = False
                layer_merged_with_num = -1
        elif "parallel_layer" in exec_stat and prev_layer_merged:
            j = layer_merged_with_num + 1
            while (j < i and exec_stats_list[j]['merged'] != True):
                layer_PEs = exec_stats_list[j]['mm_tiles']
                if required_PEs + layer_PEs <=  in_cache_arch.num_PEs:
                    overlap = True
                    exec_stats_list[j]['mm_tiles'] = required_PEs + layer_PEs
                    # Calculate the number of weight elements in merged layer [i] and add to main layer [j]
                    merged_layer_params = network_config['layers'][exec_stat['name']]
                    num_weight_elems = calc_weight_elems(merged_layer_params)
                    if "extra_weights" in exec_stats_list[j]:
                        exec_stats_list[j]['extra_weights'] += num_weight_elems
                    else:
                        exec_stats_list[j]['extra_weights'] = num_weight_elems
                    exec_stats_list[i]['merged'] = True
                    prev_layer_merged = True
                    layer_merged_with_num = j
                    #print("Merged layers child layers "+str(i)+" and "+str(j))
                    break
                else:
                    j += 1
            if (overlap == False):
                tot_exec_cycles += exec_stat['exec_cycles']
                prev_layer_merged = False
                layer_merged_with_num = -1            
        else:
            tot_exec_cycles += exec_stat['exec_cycles']

        if next_layer_read_inputs:
            tot_dram_bytes_read += exec_stat['input_bytes_read'] + exec_stat['kernel_bytes_read']
            exec_stat['input_bytes_dram'] = True
        else:
            exec_stat['input_bytes_dram'] = False
            tot_dram_bytes_read += exec_stat['kernel_bytes_read']
        
        if (exec_stat['output_bytes_written'] + exec_stat['kernel_bytes_read']) > (in_cache_config['cache_size'] * 1024 * 1024):
            tot_dram_bytes_written += exec_stat['output_bytes_written']
            next_layer_read_inputs = True
            exec_stat['output_bytes_dram'] = True
        else:
            exec_stat['output_bytes_dram'] = False
            next_layer_read_inputs = False
        
        #tot_dram_bytes_written += exec_stat['output_bytes_written']
        tot_weight_bytes += exec_stat['kernel_bytes_read']

        i += 1
    return tot_exec_cycles, tot_dram_bytes_read, tot_dram_bytes_written

def in_cache_analyze_exec_stats(exec_stats_list, network_config, in_cache_config, in_cache_arch):
    i = 0
    tot_exec_cycles = 0
    tot_dram_bytes_read = 0
    tot_dram_bytes_written = 0    
    tot_weight_bytes = 0
    next_layer_read_inputs = True
    while (i < len(exec_stats_list)):
        exec_stat = exec_stats_list[i]
        #if network_type == "object_detection" and "conv" in exec_stat['inp_layer'] :
        if exec_stat['inp_layer'] != 'prev_layer' and exec_stat['inp_layer'] != -2 :
          # Check if we can run this layer in parallel with any other following layers
            overlap = False
            required_PEs = exec_stat['mm_tiles']
            #j = int(exec_stat['inp_layer'].split("_")[-1])
            inp_layer = exec_stat['inp_layer']
            inp_layer_num = network_config['layers'][inp_layer]['num']
            j = inp_layer_num
            #while ( j < len(exec_stats_list)):
            while (j < i):
                if (exec_stats_list[j]['merged'] != True):
                    layer_PEs = exec_stats_list[j]['mm_tiles']
                    if required_PEs + layer_PEs <=  in_cache_arch.num_PEs:
                        overlap = True
                        exec_stats_list[j]['mm_tiles'] = required_PEs + layer_PEs
                        # Calculate the number of weight elements in merged layer [i] and add to main layer [j]
                        merged_layer_params = network_config['layers'][exec_stat['name']]
                        num_weight_elems = calc_weight_elems(merged_layer_params)
                        if "extra_weights" in exec_stats_list[j]:
                            exec_stats_list[j]['extra_weights'] += num_weight_elems
                        else:
                            exec_stats_list[j]['extra_weights'] = num_weight_elems
                        exec_stats_list[i]['merged'] = True
                        #print("Merged layers "+str(i)+" and "+str(j))
                        break
                    else:
                        j += 1
                else:
                    j += 1
            if (overlap == False):
                tot_exec_cycles += exec_stat['exec_cycles']
        else:
            tot_exec_cycles += exec_stat['exec_cycles']
        #print("exec cycles: ", exec_stat['exec_cycles'])
        #tot_dram_bytes_read += exec_stat['input_bytes_read'] + exec_stat['kernel_bytes_read']
        if next_layer_read_inputs:
            tot_dram_bytes_read += exec_stat['input_bytes_read'] + exec_stat['kernel_bytes_read']
            exec_stat['input_bytes_dram'] = True
        else:
            exec_stat['input_bytes_dram'] = False
            tot_dram_bytes_read += exec_stat['kernel_bytes_read']
        
        if (exec_stat['output_bytes_written'] + exec_stat['kernel_bytes_read']) > (in_cache_config['cache_size'] * 1024 * 1024):
            tot_dram_bytes_written += exec_stat['output_bytes_written']
            next_layer_read_inputs = True
            exec_stat['output_bytes_dram'] = True
        else:
            exec_stat['output_bytes_dram'] = False
            next_layer_read_inputs = False
        
        #tot_dram_bytes_written += exec_stat['output_bytes_written']
        tot_weight_bytes += exec_stat['kernel_bytes_read']

        i += 1
    print("total_weights ",tot_weight_bytes)
    return tot_exec_cycles, tot_dram_bytes_read, tot_dram_bytes_written  

def in_cache_calc_weight_load_time(exec_stats_list, network_config, in_cache_config, in_cache_arch):
    #FixMe: Naga, TO DO
    i = 1
    tot_weight_load_time = 0
    while (i < len(exec_stats_list)):
        exec_stat_cur_layer = exec_stats_list[i]
        if (exec_stat_cur_layer['mm_tiles'] != 0 and exec_stat_cur_layer['merged'] == False):
            exec_stat_prev_layer = exec_stats_list[i-1]
            num_weight_elems = calc_weight_elems(network_config['layers'][exec_stat_cur_layer['name']])
            if "extra_weights" in exec_stat_cur_layer:
                num_weight_elems += exec_stat_cur_layer['extra_weights']
            weight_read_time = num_weight_elems / (in_cache_config['offchip_bw'] * 1e+9) #assuming 8-bit precision 
            #elems_per_cb_row = (cb_config['array_dims'][1] * cb_config['cell_precsn']) / 8
            #weight_write_time = (num_weight_elems / elems_per_cb_row) * float(cb_config['write_time'])
            #weight_load_time = max(weight_read_time, weight_write_time)
            weight_load_time = weight_read_time
            prev_layer_exec_time = exec_stat_prev_layer['exec_cycles'] * float(in_cache_config['comp_time'])
            if exec_stat_cur_layer['mm_tiles'] + exec_stat_prev_layer['mm_tiles'] <= in_cache_arch.num_PEs:
                if prev_layer_exec_time < weight_load_time:
                    tot_weight_load_time += (weight_load_time - prev_layer_exec_time)
            else:
                tot_weight_load_time += weight_load_time

        i += 1
    #print("Weight load time: ", tot_weight_load_time)
    return tot_weight_load_time

def calc_weight_elems(layer_params):
    num_weight_elems = 0
    layer_type = layer_params['type']
    if "conv" in layer_type: 
        num_weight_elems += np.prod(layer_params['kernel'])
    elif "FC" in layer_type:
        num_weight_elems += np.prod(layer_params['kernel'])
    elif "add" or "pool" in layer_type:
        num_weight_elems = 0
    else: 
        print(layer_type+" not implemented")
        sys.exit()
    
    return num_weight_elems

        
        