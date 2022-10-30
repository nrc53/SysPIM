import numpy as np
import math
import sys

# crossbar implementation for various layers
class crossbar_arch:
    def __init__(self, cb_config, comp_precsn, comp_mode):
        self.cb_config = cb_config
        self.comp_precsn = comp_precsn
        self.comp_mode = comp_mode
        cb_rows = cb_config['array_dims'][0]
        cb_cols = cb_config['array_dims'][1]
        cb_cell_precsn = cb_config['cell_precsn']
        self.elems_per_cb = (cb_cols * cb_cell_precsn * cb_rows ) / 8
        self.rows_actv_per_cycle = {3 : {1: {1:7, 2:2}, 2: {1:2 , 2:1 } }, \
            4 : {1: {1:15, 2:5}, 2: {1:5 , 2:1 }}, \
            5 : {1: {1:31, 2:10}, 2: {1:10 , 2:3}}, \
            6 : {1: {1:63, 2:21}, 2: {1:21 , 2:7}}} 

    def conv_depthwise(self, input_dims, layer_params):
        #print("input_dims: ", input_dims, " layer_params: ", layer_params)
        cb_rows = self.cb_config['array_dims'][0]
        cb_cols = self.cb_config['array_dims'][1]
        cb_cell_precsn = self.cb_config['cell_precsn']
        elemns_per_row = (cb_cols * cb_cell_precsn) / self.comp_precsn  
        kernel_dims = layer_params['kernel']
        operands_per_mac = np.prod(kernel_dims[:-1])
        padding = layer_params['padding']  
        output_dim_r = round((input_dims[0] - kernel_dims[0] + padding) / layer_params['stride']) + 1
        output_dim_c = round((input_dims[1] - kernel_dims[1] + padding) / layer_params['stride']) + 1
        tot_num_macs = output_dim_r * output_dim_c * kernel_dims[-1]
        if 'row_activations' in self.cb_config:
            rows_per_cycle = self.cb_config['row_activations']
        else:
            rows_per_cycle = self.rows_actv_per_cycle[self.cb_config['adc_precsn']][self.cb_config['dac_precsn']][cb_cell_precsn]
        output_dims = [output_dim_r, output_dim_c, kernel_dims[-1]]
        if self.comp_mode == "max_throughput":
            if cb_rows > operands_per_mac:
                num_macs_with_common_inp = math.floor(cb_rows / operands_per_mac)
            else:
                num_macs_with_common_inp = 1
            cycles_per_mac = math.ceil(operands_per_mac / rows_per_cycle) * (self.comp_precsn / self.cb_config['dac_precsn'])
            tot_cycles = cycles_per_mac * math.ceil(tot_num_macs / num_macs_with_common_inp)
            num_cbs = math.ceil(kernel_dims[-1] / elemns_per_row)
        elif self.comp_mode == "area_efficient":
            cycles_per_mac = math.ceil(operands_per_mac / rows_per_cycle) * (self.comp_precsn / self.cb_config['dac_precsn'])
            num_macs_with_common_inp = 1
            tot_cycles = cycles_per_mac * math.ceil(tot_num_macs / num_macs_with_common_inp)
            kernels_per_col = math.floor(cb_rows / operands_per_mac)    
            num_cbs = math.ceil(kernel_dims[-1] / (elemns_per_row * kernels_per_col))   
        # if operands per mac require more than one crossbar array, we can run the multiple arrays in parallel
        parallel_arrays_per_mac = math.ceil(operands_per_mac / cb_rows)
        #print("Parallel arrays: ", parallel_arrays_per_mac)
        tot_cycles = math.ceil(tot_cycles/parallel_arrays_per_mac)
        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)

        absolute_macs = math.ceil(operands_per_mac / 2) * tot_num_macs
        
        exec_stats = {'num_cb':num_cbs, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, 'absolute_macs':absolute_macs}
        return exec_stats

    def conv(self, input_dims, layer_params):
        #print("input_dims: ", input_dims, " layer_params: ", layer_params)
        cb_rows = self.cb_config['array_dims'][0]
        cb_cols = self.cb_config['array_dims'][1]
        cb_cell_precsn = self.cb_config['cell_precsn']
        elemns_per_row = (cb_cols * cb_cell_precsn) / self.comp_precsn  
        kernel_dims = layer_params['kernel']
        operands_per_mac = np.prod(kernel_dims[:-1])
        #print("operands per mac: ", operands_per_mac)
        padding = layer_params['padding']  
        output_dim_r = round((input_dims[0] - kernel_dims[0] + padding) / layer_params['stride']) + 1
        output_dim_c = round((input_dims[1] - kernel_dims[1] + padding) / layer_params['stride']) + 1
        #print(opt_dim)
        #sys.exit()
        num_macs_with_common_inp = kernel_dims[-1]
        #print("num_macs_with_common_inp: ", num_macs_with_common_inp)
        tot_num_macs = num_macs_with_common_inp * output_dim_r * output_dim_c
        if 'row_activations' in self.cb_config:
            rows_per_cycle = self.cb_config['row_activations']
        else:
            rows_per_cycle = self.rows_actv_per_cycle[self.cb_config['adc_precsn']][self.cb_config['dac_precsn']][cb_cell_precsn]
        #print("rows accumulated per cycle: ", rows_per_cycle)
        output_dims = [output_dim_r, output_dim_c, num_macs_with_common_inp]
        if self.comp_mode == 'max_throughput':
            if operands_per_mac <= cb_rows:
                num_cbs = math.ceil(num_macs_with_common_inp / elemns_per_row)
            else:
                num_cbs = math.ceil(num_macs_with_common_inp / elemns_per_row) * math.ceil(operands_per_mac / cb_rows)
            cycles_per_mac = math.ceil(operands_per_mac / rows_per_cycle) * (self.comp_precsn / self.cb_config['dac_precsn'])
            #print("cycles per mac: ", cycles_per_mac)
            tot_cycles = cycles_per_mac * output_dim_c * output_dim_r
            #print("tot_cycles: ", tot_cycles)
            #sys.exit()
        elif self.comp_mode == 'area_efficient':
            # we consider all number of filters that can be fit into a crossbar column completely (cb_rows = 128, filter_size = 9, num_filters = 14)
            if operands_per_mac <= cb_rows:
                kernels_per_col = math.floor(cb_rows / operands_per_mac)
                num_cbs = math.ceil(math.ceil(num_macs_with_common_inp / elemns_per_row) / kernels_per_col)
            else:
                kernels_per_col = 1
                num_cbs = math.ceil(num_macs_with_common_inp / elemns_per_row) * math.ceil(operands_per_mac / cb_rows)
            cycles_per_mac = math.ceil(operands_per_mac / rows_per_cycle) * (self.comp_precsn / self.cb_config['dac_precsn'])            
            tot_cycles = cycles_per_mac * output_dim_r * kernels_per_col * output_dim_c # if 2 or more filters fit into a single corssbar column, in one cycle we can only execute one kernel

        # if operands per mac require more than one crossbar array, we can run the multiple arrays in parallel
        parallel_arrays_per_mac = math.ceil(operands_per_mac / cb_rows)
        #print("Parallel arrays: ", parallel_arrays_per_mac)
        tot_cycles = math.ceil(tot_cycles/parallel_arrays_per_mac)
        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)

        absolute_macs = math.ceil(operands_per_mac / 2) * tot_num_macs
        
        exec_stats = {'num_cb':num_cbs, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, 'absolute_macs':absolute_macs}
        return exec_stats

    def fc(self, input_dims, layer_params):
        input_dims = [1, np.prod(input_dims)]
        cb_rows = self.cb_config['array_dims'][0]
        cb_cols = self.cb_config['array_dims'][1]
        cb_cell_precsn = self.cb_config['cell_precsn']
        elemns_per_row = (cb_cols * cb_cell_precsn) / self.comp_precsn  
        kernel_dims = layer_params['kernel']
        num_wt_elems = kernel_dims[0] * kernel_dims[1] 
        operands_per_mac = kernel_dims[0]
        tot_num_macs = kernel_dims[1] * input_dims [0]
        if 'row_activations' in self.cb_config:
            rows_per_cycle = self.cb_config['row_activations']
        else:
            rows_per_cycle = self.rows_actv_per_cycle[self.cb_config['adc_precsn']][self.cb_config['dac_precsn']][cb_cell_precsn]
        cycles_per_mac = math.ceil(operands_per_mac / rows_per_cycle) * (self.comp_precsn / self.cb_config['dac_precsn']) 
        num_macs_with_common_inp = kernel_dims[1]
        tot_cycles = cycles_per_mac * (tot_num_macs / num_macs_with_common_inp) * input_dims[0]
        num_cbs = math.ceil(num_wt_elems / (elemns_per_row * cb_rows))
        output_dims = [input_dims[0], kernel_dims[1]]
        # if operands per mac require more than one crossbar array, we can run the multiple arrays in parallel
        parallel_arrays_per_mac = math.ceil(operands_per_mac / cb_rows)
        #print("Parallel arrays: ", parallel_arrays_per_mac)
        tot_cycles = math.ceil(tot_cycles/parallel_arrays_per_mac)
        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)

        absolute_macs = math.ceil(operands_per_mac / 2) * tot_num_macs
        
        exec_stats = {'num_cb':num_cbs, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read, 'absolute_macs':absolute_macs}
        return exec_stats

    def relu(self, input_dims):
        num_alus = self.cb_config['num_alus']
        tot_num_ops = np.prod(input_dims)
        tot_cycles = math.ceil(tot_num_ops / num_alus)
        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(input_dims)
        return tot_cycles
        # exec_stats = {'exec_cycles':tot_cycles, 'output_dims':input_dims, 'input_bytes_read':inp_bytes_read, \
        #     'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}

        # return exec_stats

    def residual_add(self, input_dims):
        num_alus = self.cb_config['num_alus']
        tot_num_ops = input_dims[0] * input_dims[1] * input_dims[2]
        tot_cycles = math.ceil(tot_num_ops / num_alus)
        inp_bytes_read = np.prod(input_dims) * 2
        output_bytes_written = np.prod(input_dims)
        exec_stats = {'num_cb':0, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':input_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def pool(self, input_dims, layer_params):
        num_alus = self.cb_config['num_alus']
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
        exec_stats = {'num_cb':0, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def concat(self, input_dims):
        concat_dim = 0
        for dim in input_dims:
            concat_dim += dim[-1]
        output_dims = [input_dims[0][0], input_dims[0][1], concat_dim]
        exec_stats = {'num_cb':0, 'exec_cycles':0, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':0, \
            'output_bytes_written':0, 'kernel_bytes_read':0}
        return exec_stats

    def elem_wise_mult(self, input_dims):
        tot_mult_ops = input_dims[0] + input_dims[1] # Specific to the LSTM implementation
        num_alus = self.cb_config['num_alus']
        tot_cycles = math.ceil(tot_mult_ops / num_alus)
        input_bytes_read = tot_mult_ops * 2
        output_bytes_written = input_dims[0]
        exec_stats = {'num_cb':0, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':[input_dims[0]], 'input_bytes_read':input_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def elem_wise_add(self, input_dims):
        tot_add_ops = input_dims[0]  # Specific to the LSTM implementation
        num_alus = self.cb_config['num_alus']
        input_bytes_read = tot_add_ops * 2
        tot_cycles = math.ceil(tot_add_ops / num_alus)
        output_bytes_written = input_dims[0]
        exec_stats = {'num_cb':0, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':[input_dims[0]], 'input_bytes_read':input_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats
        
def cb_analyze_exec_stats_parallel_branches(exec_stats_list, network_config, cb_config):
    i = 0
    tot_exec_cycles = 0
    tot_dram_bytes_read = 0
    tot_dram_bytes_written = 0
    prev_layer_merged = False
    layer_merged_with_num = -1
    while (i < len(exec_stats_list)):
        exec_stat = exec_stats_list[i]
        overlap = False
        required_cbs = exec_stat['num_cb']
        #print(i, exec_stat, prev_layer_merged)
        # Check if first node/layer in the parallel (second) branch can be merged 
        if exec_stat['inp_layer'] != "prev_layer" and exec_stat['type'] != "concat":
            first_branch_layer = exec_stat['parallel_layer']
            first_branch_layer_num = network_config['layers'][first_branch_layer]['num']
            j = first_branch_layer_num
            while (j < i and exec_stats_list[j]['merged'] != True):
                layer_cbs = exec_stats_list[j]['num_cb']
                if required_cbs + layer_cbs <=  cb_config['array_num']:
                    overlap = True
                    exec_stats_list[j]['num_cb'] = required_cbs + layer_cbs
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
                layer_cbs = exec_stats_list[j]['num_cb']
                if required_cbs + layer_cbs <=  cb_config['array_num']:
                    overlap = True
                    exec_stats_list[j]['num_cb'] = required_cbs + layer_cbs
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
        
        #tot_dram_bytes_read += exec_stat['input_bytes_read'] + exec_stat['kernel_bytes_read']
        tot_dram_bytes_read += exec_stat['input_bytes_read'] #+ exec_stat['kernel_bytes_read']
        tot_dram_bytes_written += exec_stat['output_bytes_written']

        i += 1
    return tot_exec_cycles, tot_dram_bytes_read, tot_dram_bytes_written

def cb_analyze_exec_stats(exec_stats_list, network_config, cb_config):
    i = 0
    tot_exec_cycles = 0
    tot_dram_bytes_read = 0
    tot_dram_bytes_written = 0    
    tot_weight_bytes = 0
    while (i < len(exec_stats_list)):
        exec_stat = exec_stats_list[i]
        #if network_type == "object_detection" and "conv" in exec_stat['inp_layer'] :
        if exec_stat['inp_layer'] != 'prev_layer' and exec_stat['inp_layer'] != -2 :
          # Check if we can run this layer in parallel with any other following layers
            overlap = False
            required_cbs = exec_stat['num_cb']
            #j = int(exec_stat['inp_layer'].split("_")[-1])
            inp_layer = exec_stat['inp_layer']
            inp_layer_num = network_config['layers'][inp_layer]['num']
            j = inp_layer_num
            #while ( j < len(exec_stats_list)):
            while (j < i):
                if (exec_stats_list[j]['merged'] != True):
                    layer_cbs = exec_stats_list[j]['num_cb']
                    if required_cbs + layer_cbs <=  cb_config['array_num']:
                        overlap = True
                        exec_stats_list[j]['num_cb'] = required_cbs + layer_cbs
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

        tot_dram_bytes_read += exec_stat['input_bytes_read']  #+ exec_stat['kernel_bytes_read']
        tot_dram_bytes_written += exec_stat['output_bytes_written']
        tot_weight_bytes += exec_stat['kernel_bytes_read']

        i += 1
    print("total_weights ",tot_weight_bytes)
    return tot_exec_cycles, tot_dram_bytes_read, tot_dram_bytes_written  

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

def calc_cb_capacity(cb_config):
    cb_rows = cb_config['array_dims'][0]
    cb_cols = cb_config['array_dims'][1]
    cb_cell_precsn = cb_config['cell_precsn']
    elems_per_cb = (cb_cols * cb_cell_precsn * cb_rows ) / 8  #FixMe: Naga, assuming inference precision as 8, make it a parameter
    cb_capacity = elems_per_cb * cb_config['array_num']

    return cb_capacity

def cb_calc_weight_load_time(exec_stats_list, network_config, cb_config):
    #FixMe: Naga, TO DO
    i = 1
    tot_weight_load_time = 0
    while (i < len(exec_stats_list)):
        exec_stat_cur_layer = exec_stats_list[i]
        if (exec_stat_cur_layer['num_cb'] != 0 and exec_stat_cur_layer['merged'] == False):
            exec_stat_prev_layer = exec_stats_list[i-1]
            num_weight_elems = calc_weight_elems(network_config['layers'][exec_stat_cur_layer['name']])
            if "extra_weights" in exec_stat_cur_layer:
                num_weight_elems += exec_stat_cur_layer['extra_weights']
            weight_read_time = num_weight_elems / (cb_config['offchip_bw'] * 1e+9) #assuming 8-bit precision 
            elems_per_cb_row = (cb_config['array_dims'][1] * cb_config['cell_precsn']) / 8
            #weight_write_time = (num_weight_elems / elems_per_cb_row) * float(cb_config['write_time'])
            #weight_load_time = max(weight_read_time, weight_write_time)
            weight_load_time = weight_read_time
            prev_layer_exec_time = exec_stat_prev_layer['exec_cycles'] * float(cb_config['mac_time'])
            if exec_stat_cur_layer['num_cb'] + exec_stat_prev_layer['num_cb'] <= cb_config['array_num']:
                if prev_layer_exec_time < weight_load_time:
                    tot_weight_load_time += (weight_load_time - prev_layer_exec_time)
            else:
                tot_weight_load_time += weight_load_time

        i += 1
    #print("Weight load time: ", tot_weight_load_time)
    return tot_weight_load_time