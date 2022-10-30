import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import crossbar_layers
import in_cache_layers
import in_dram_layers
import crossbar_power
import in_cache_power
import in_dram_power

def neural_net_inference(pim_hardware, hardware_config, dataflow, network, inp_dims):
    if pim_hardware == "crossbar":
        tot_time, bytes_read, bytes_written, stats, energy_stats = cb_inference(hardware_config, dataflow, network, inp_dims)
    elif pim_hardware == "in_cache":
        tot_time, bytes_read, bytes_written, stats, energy_stats = in_cache_inference(hardware_config, dataflow, network, inp_dims)
    elif pim_hardware == "in_dram":
        tot_time, bytes_read, bytes_written, stats, energy_stats = in_dram_inference(hardware_config, dataflow, network, inp_dims)
    else:
        print(pim_hardware+" not implemented!")
        sys.exit()
    return tot_time, bytes_read, bytes_written, stats, energy_stats

def in_dram_inference(config, dataflow, network,  input_dims = None):
    network_file = open(network+".yaml", 'r')
    network_config = yaml.load(network_file, Loader=yaml.FullLoader)
    in_dram_file = open(config+".yaml", 'r')
    in_dram_config = yaml.load(in_dram_file, Loader=yaml.FullLoader)
    in_dram_arch = in_dram_layers.in_dram_arch(in_dram_config, 8, "max_throughput")
    in_dram_power_model = in_dram_power.in_dram_power(in_dram_config)
    #in_dram_power_model = crossbar_power.crossbar_power(cb_config)
    inp_dims = input_dims
    exec_stats = {}
    inp_dims_list = []
    inp_dims_list.append(inp_dims)
    i = 0
    for layer in network_config['layers']:
        layer_params = network_config['layers'][layer]
        layer_type = layer_params['type']
        #print(layer_params)
        layer_exec_stats = {}
        if layer_type == 'conv_normal':
            layer_exec_stats = in_dram_arch.conv(inp_dims, layer_params)
        elif layer_type == 'conv_dw':
            layer_exec_stats = in_dram_arch.conv_depthwise(inp_dims, layer_params)
        elif layer_type == 'fc':
            if network_config['network_type'] == "lstm":
                inp_dims = layer_params['input_dims']
            layer_exec_stats = in_dram_arch.fc(inp_dims, layer_params)
        elif layer_type == 'pool':
            layer_exec_stats = in_dram_arch.pool(inp_dims, layer_params)  
        elif layer_type == 'residual_add':
            layer_exec_stats = in_dram_arch.residual_add(inp_dims)
        elif layer_type == 'concat':
            # Find the output dims of all the input layers/nodes to concat layer (open_pose network)
            input_layers = layer_params['input']
            inp_dims_concat = []
            for layer_name in input_layers:
                layer_num = network_config['layers'][layer_name]['num']
                inp_dims_concat.append(inp_dims_list[layer_num])
            layer_exec_stats = in_dram_arch.concat(inp_dims_concat)
        elif layer_type == "elem_wise_add":
            # Special layer for LSTM, input_dims are pre-populated in yaml files
            inp_dims = layer_params['input_dims']
            layer_exec_stats = in_dram_arch.elem_wise_add(inp_dims)
        elif layer_type == "elem_wise_mult":
            # Special layer for LSTM, input_dims are pre-populated in yaml files
            inp_dims = layer_params['input_dims']
            layer_exec_stats = in_dram_arch.elem_wise_mult(inp_dims)
        else:
            print(layer_type+" not implemented")
            sys.exit()    
        layer_exec_stats['inp_layer'] = layer_params['input']
        layer_exec_stats['type'] = layer_params['type']
        layer_exec_stats['name'] = layer
        if "parallel_op" in layer_params:
            layer_exec_stats['parallel_layer'] = layer_params['parallel_op']
        
        exec_stats[layer] = layer_exec_stats
        #exec_stats.append(layer_exec_stats)
        i += 1
        inp_dims = layer_exec_stats['output_dims']     
        inp_dims_list.append(inp_dims)
        
    #collect the stats in dict for the output report
    stats_dict = {}
    d_layer_name = []
    d_in_dram_num = []
    d_exec_cycles = []
    d_input_dims = []
    d_output_dims = []
    d_input_bytes_read = []
    d_kernel_bytes_read = []
    d_output_bytes_written = []
    d_output_dims= []

    exec_cycles = 0
    bytes_read = 0
    bytes_written = 0
    exec_stats_list = []
    for stats in exec_stats:
        exec_stats[stats]['merged'] = False
        d_layer_name.append(exec_stats[stats]['name'])
        d_in_dram_num.append(exec_stats[stats]['num_banks'])
        d_exec_cycles.append(exec_stats[stats]['exec_cycles'])
        d_input_dims.append(exec_stats[stats]['input_dims'])
        d_output_dims.append(exec_stats[stats]['output_dims'])
        d_input_bytes_read.append(exec_stats[stats]['input_bytes_read'])
        d_kernel_bytes_read.append(exec_stats[stats]['kernel_bytes_read'])
        d_output_bytes_written.append(exec_stats[stats]['output_bytes_written'])
        exec_cycles += exec_stats[stats]['exec_cycles']
        bytes_read += exec_stats[stats]['input_bytes_read']
        bytes_read += exec_stats[stats]['kernel_bytes_read']
        bytes_written += exec_stats[stats]['output_bytes_written']
        exec_stats_list.append(exec_stats[stats])
    stats_dict['layer_name'] = d_layer_name
    stats_dict['num_cbs'] = d_in_dram_num
    stats_dict['exec_cycles'] = d_exec_cycles
    stats_dict['input_dims'] = d_input_dims
    stats_dict['output_dims'] = d_output_dims
    stats_dict['input_bytes_read'] = d_input_bytes_read
    stats_dict['kernel_bytes_read'] = d_kernel_bytes_read
    stats_dict['output_bytes_written'] = d_output_bytes_written
    exec_stats_list = []
    for stats in exec_stats:
        exec_stats[stats]['merged'] = False
        exec_stats_list.append(exec_stats[stats])

    #print("exec_cycles: ", exec_cycles)
    tot_time = exec_cycles * float(in_dram_config['cycle_time']) 
    energy_stats_df = in_dram_power_model.calc_energy(exec_stats_list)

    stats_dict_df = pd.DataFrame(stats_dict)
    return tot_time, bytes_read, bytes_written, stats_dict_df, energy_stats_df

    #print(exec_stats_list)

def cb_inference(config, dataflow, network,  input_dims = None):
    network_file = open(network+".yaml", 'r')
    network_config = yaml.load(network_file, Loader=yaml.FullLoader)
    cb_file = open(config+".yaml", 'r')
    cb_config = yaml.load(cb_file, Loader=yaml.FullLoader)
    cb_arch = crossbar_layers.crossbar_arch(cb_config, 8, "max_throughput")
    cb_power_model = crossbar_power.crossbar_power(cb_config)
    #cb_arch = layers.crossbar_arch(cb_config, 8, "area_efficient")
    inp_dims = input_dims
    exec_stats = {}
    inp_dims_list = []
    inp_dims_list.append(inp_dims)
    i = 0
    for layer in network_config['layers']:
        layer_params = network_config['layers'][layer]
        layer_type = layer_params['type']
        #print(layer_params)
        layer_exec_stats = {}
        if layer_type == 'conv_normal':
            layer_exec_stats = cb_arch.conv(inp_dims, layer_params)
        elif layer_type == 'conv_dw':
            layer_exec_stats = cb_arch.conv_depthwise(inp_dims, layer_params)
        elif layer_type == 'fc':
            if network_config['network_type'] == "lstm":
                inp_dims = layer_params['input_dims']
            layer_exec_stats = cb_arch.fc(inp_dims, layer_params)
        elif layer_type == 'pool':
            layer_exec_stats = cb_arch.pool(inp_dims, layer_params)  
        elif layer_type == 'residual_add':
            layer_exec_stats = cb_arch.residual_add(inp_dims)
        elif layer_type == 'concat':
            # Find the output dims of all the input layers/nodes to concat layer (open_pose network)
            input_layers = layer_params['input']
            inp_dims_concat = []
            for layer_name in input_layers:
                layer_num = network_config['layers'][layer_name]['num']
                inp_dims_concat.append(inp_dims_list[layer_num])
            layer_exec_stats = cb_arch.concat(inp_dims_concat)
        elif layer_type == "elem_wise_add":
            # Special layer for LSTM, input_dims are pre-populated in yaml files
            inp_dims = layer_params['input_dims']
            layer_exec_stats = cb_arch.elem_wise_add(inp_dims)
        elif layer_type == "elem_wise_mult":
            # Special layer for LSTM, input_dims are pre-populated in yaml files
            inp_dims = layer_params['input_dims']
            layer_exec_stats = cb_arch.elem_wise_mult(inp_dims)
        else:
            print(layer_type+" not implemented")
            sys.exit()    
        # if network_config['network_type'] == "object_detection":
        #     layer_exec_stats['inp_layer'] = layer_params['input']
        # else:
        #     layer_exec_stats['inp_layer'] = "prev_layer"
        layer_exec_stats['inp_layer'] = layer_params['input']
        layer_exec_stats['type'] = layer_params['type']
        layer_exec_stats['name'] = layer
        if "parallel_op" in layer_params:
            layer_exec_stats['parallel_layer'] = layer_params['parallel_op']
        
        exec_stats[layer] = layer_exec_stats
        if "absolute_macs" in layer_exec_stats:
            print(layer_exec_stats)

        #exec_stats.append(layer_exec_stats)
        i += 1
        inp_dims = layer_exec_stats['output_dims']     
        inp_dims_list.append(inp_dims)
        
    #collect the stats in dict for the output report
    stats_dict = {}
    d_layer_name = []
    d_cb_num = []
    d_exec_cycles = []
    d_input_dims = []
    d_output_dims = []
    d_input_bytes_read = []
    d_kernel_bytes_read = []
    d_output_bytes_written = []
    d_output_dims= []

    exec_stats_list = []
    for stats in exec_stats:
        exec_stats[stats]['merged'] = False
        d_layer_name.append(exec_stats[stats]['name'])
        d_cb_num.append(exec_stats[stats]['num_cb'])
        d_exec_cycles.append(exec_stats[stats]['exec_cycles'])
        d_input_dims.append(exec_stats[stats]['input_dims'])
        d_output_dims.append(exec_stats[stats]['output_dims'])
        d_input_bytes_read.append(exec_stats[stats]['input_bytes_read'])
        d_kernel_bytes_read.append(exec_stats[stats]['kernel_bytes_read'])
        d_output_bytes_written.append(exec_stats[stats]['output_bytes_written'])
        exec_stats_list.append(exec_stats[stats])
    stats_dict['layer_name'] = d_layer_name
    stats_dict['num_cbs'] = d_cb_num
    stats_dict['exec_cycles'] = d_exec_cycles
    stats_dict['input_dims'] = d_input_dims
    stats_dict['output_dims'] = d_output_dims
    stats_dict['input_bytes_read'] = d_input_bytes_read
    stats_dict['kernel_bytes_read'] = d_kernel_bytes_read
    stats_dict['output_bytes_written'] = d_output_bytes_written
    exec_stats_list = []
    for stats in exec_stats:
        exec_stats[stats]['merged'] = False
        exec_stats_list.append(exec_stats[stats])

    if network_config['network_type'] != 'pose_estimation':
        exec_cycles, bytes_read, bytes_written = crossbar_layers.cb_analyze_exec_stats(exec_stats_list, network_config, cb_config)
    else:
        exec_cycles, bytes_read, bytes_written = crossbar_layers.cb_analyze_exec_stats_parallel_branches(exec_stats_list, network_config, cb_config)

    weight_load_time = crossbar_layers.cb_calc_weight_load_time(exec_stats_list, network_config, cb_config)
    #print("exec_cycles: ", exec_cycles)
    tot_time = exec_cycles * float(cb_config['mac_time']) + weight_load_time
    energy_stats_df = cb_power_model.calc_energy(exec_stats_list)

    stats_dict_df = pd.DataFrame(stats_dict)
    return tot_time, bytes_read, bytes_written, stats_dict_df, energy_stats_df

    #print(exec_stats_list)

def in_cache_inference(config, dataflow, network,  input_dims = None):
    network_file = open(network+".yaml", 'r')
    network_config = yaml.load(network_file, Loader=yaml.FullLoader)
    in_cache_file = open(config+".yaml", 'r')
    in_cache_config = yaml.load(in_cache_file, Loader=yaml.FullLoader)
    in_cache_arch = in_cache_layers.in_cache_arch(in_cache_config, 8, "max_throughput")
    in_cache_power_model = in_cache_power.in_cache_power(in_cache_config)
    #in_cache_arch = layers.crossbar_arch(in_cache_config, 8, "area_efficient")
    inp_dims = input_dims
    exec_stats = {}
    inp_dims_list = []
    inp_dims_list.append(inp_dims)
    i = 0
    for layer in network_config['layers']:
        layer_params = network_config['layers'][layer]
        layer_type = layer_params['type']
        #print(" layer in main ", layer, layer_params)
        #print(layer_params)
        layer_exec_stats = {}
        if layer_type == 'conv_normal':
            layer_exec_stats = in_cache_arch.conv(inp_dims, layer_params)
        elif layer_type == 'conv_dw':
            layer_exec_stats = in_cache_arch.conv_depthwise(inp_dims, layer_params)
        elif layer_type == 'fc':
            if network_config['network_type'] == "lstm":
                inp_dims = layer_params['input_dims']
            layer_exec_stats = in_cache_arch.fc(inp_dims, layer_params)
        elif layer_type == 'pool':
            layer_exec_stats = in_cache_arch.pool(inp_dims, layer_params)  
        elif layer_type == 'residual_add':
            layer_exec_stats = in_cache_arch.residual_add(inp_dims)
        elif layer_type == 'concat':
            # Find the output dims of all the input layers/nodes to concat layer (open_pose network)
            input_layers = layer_params['input']
            inp_dims_concat = []
            for layer_name in input_layers:
                layer_num = network_config['layers'][layer_name]['num']
                inp_dims_concat.append(inp_dims_list[layer_num])
            layer_exec_stats = in_cache_arch.concat(inp_dims_concat)
        elif layer_type == "elem_wise_add":
            # Special layer for LSTM, input_dims are pre-populated in yaml files
            inp_dims = layer_params['input_dims']
            layer_exec_stats = in_cache_arch.elem_wise_add(inp_dims)
        elif layer_type == "elem_wise_mult":
            # Special layer for LSTM, input_dims are pre-populated in yaml files
            inp_dims = layer_params['input_dims']
            layer_exec_stats = in_cache_arch.elem_wise_mult(inp_dims)
        else:
            print(layer_type+" not implemented")
            sys.exit()    
        # if network_config['network_type'] == "object_detection":
        #     layer_exec_stats['inp_layer'] = layer_params['input']
        # else:
        #     layer_exec_stats['inp_layer'] = "prev_layer"
        layer_exec_stats['inp_layer'] = layer_params['input']
        layer_exec_stats['type'] = layer_params['type']
        layer_exec_stats['name'] = layer
        if "parallel_op" in layer_params:
            layer_exec_stats['parallel_layer'] = layer_params['parallel_op']
        
        exec_stats[layer] = layer_exec_stats
        #print("layer_name ", layer)
        #exec_stats.append(layer_exec_stats)
        i += 1
        inp_dims = layer_exec_stats['output_dims']     
        inp_dims_list.append(inp_dims)
        
    #sys.exit()
    #collect the stats in dict for the output report
    stats_dict = {}
    d_layer_name = []
    d_mm_tiles = []
    d_exec_cycles = []
    d_input_dims = []
    d_output_dims = []
    d_input_bytes_read = []
    d_kernel_bytes_read = []
    d_output_bytes_written = []
    d_output_dims= []

    exec_stats_list = []
    for stats in exec_stats:
        exec_stats[stats]['merged'] = False
        d_layer_name.append(exec_stats[stats]['name'])
        d_mm_tiles.append(exec_stats[stats]['mm_tiles'])
        d_exec_cycles.append(exec_stats[stats]['exec_cycles'])
        d_input_dims.append(exec_stats[stats]['input_dims'])
        d_output_dims.append(exec_stats[stats]['output_dims'])
        d_input_bytes_read.append(exec_stats[stats]['input_bytes_read'])
        d_kernel_bytes_read.append(exec_stats[stats]['kernel_bytes_read'])
        d_output_bytes_written.append(exec_stats[stats]['output_bytes_written'])
        exec_stats_list.append(exec_stats[stats])
    stats_dict['layer_name'] = d_layer_name
    stats_dict['mm_tiles'] = d_mm_tiles
    stats_dict['exec_cycles'] = d_exec_cycles
    stats_dict['input_dims'] = d_input_dims
    stats_dict['output_dims'] = d_output_dims
    stats_dict['input_bytes_read'] = d_input_bytes_read
    stats_dict['kernel_bytes_read'] = d_kernel_bytes_read
    stats_dict['output_bytes_written'] = d_output_bytes_written

    if network_config['network_type'] != 'pose_estimation':
        exec_cycles, bytes_read, bytes_written = in_cache_layers.in_cache_analyze_exec_stats(exec_stats_list, network_config, in_cache_config, in_cache_arch)
    else:
        exec_cycles, bytes_read, bytes_written = in_cache_layers.in_cache_analyze_exec_stats_parallel_branches(exec_stats_list, network_config, in_cache_config, in_cache_arch)

    weight_load_time = in_cache_layers.in_cache_calc_weight_load_time(exec_stats_list, network_config, in_cache_config, in_cache_arch)
    #print("exec_cycles: ", exec_cycles)
    tot_time = exec_cycles * float(in_cache_config['comp_time']) + weight_load_time
    stats_dict_df = pd.DataFrame(stats_dict)
    energy_stats_df = in_cache_power_model.calc_energy(exec_stats_list)
    return tot_time, bytes_read, bytes_written, stats_dict_df, energy_stats_df

    #print(exec_stats_list)

def inference(args):
    pipeline_file = open(args.pipeline+".yaml", 'r')
    pipeline_config = yaml.load(pipeline_file, Loader=yaml.FullLoader) 
    inference_stats = []
    nn_detailed_stats = {}
    nn_energy_stats = {}
    d_component_name = []
    d_exec_time = []
    d_dram_bytes_written = []
    d_dram_bytes_read = []

    for component in pipeline_config['components']:
        component_params = pipeline_config['components'][component]
        if component_params['exec_engine'] == 'cpu':
            num_occurence = component_params['occurences']
            exec_time = num_occurence * float(component_params['exec_time'])
            if 'output_dim' in component_params:
                bytes_written = (np.prod(component_params['output_dim']) * num_occurence) / (1024*1204) # assuming 8-bit precision
            inference_stats.append({'name':component, 'exec_time(sec)':exec_time, 'dram_bytes_written':bytes_written, 'dram_bytes_read':0})
            d_component_name.append(component)
            d_exec_time.append(exec_time)
            d_dram_bytes_written.append(bytes_written)
            d_dram_bytes_read.append(0)
        elif component_params['exec_engine'] == 'pim_accelerator':
            #def neural_net_inference(pim_hardware, hardware_config, dataflow, network, inp_dims):
            num_occurence = component_params['occurences']
            exec_time, bytes_read, bytes_written, stats_dict, energy_stats_df = neural_net_inference(args.pim_accelerator, args.accelerator_config, args.accelerator_dataflow, \
                component_params['neural_network'], component_params['inp_dim'])
            nn_detailed_stats[component_params['neural_network']] = stats_dict
            nn_energy_stats[component_params['neural_network']] = energy_stats_df
            exec_time = num_occurence * exec_time
            bytes_read = (num_occurence * bytes_read) / (1024*1204)
            bytes_written = (num_occurence * bytes_written) / (1024*1204)
            inference_stats.append({'name':component, 'exec_time(sec)':exec_time, 'dram_bytes_written':bytes_written, 'dram_bytes_read':bytes_read})
            d_component_name.append(component)
            d_exec_time.append(exec_time)
            d_dram_bytes_written.append(bytes_written)
            d_dram_bytes_read.append(bytes_read)        
        else:
            print(component_params['exec_engine']+" not implemented")
            sys.exit()
    
    summary_stats = {}
    summary_stats['name'] = d_component_name
    summary_stats['exec_time'] = d_exec_time
    summary_stats['dram_bytes_written'] = d_dram_bytes_written
    summary_stats['dram_bytes_read'] = d_dram_bytes_read
    summary_df = pd.DataFrame(summary_stats)
    report_file = args.report_dir+"/"+args.report_prefix+"_pim_report.xlsx"
    with pd.ExcelWriter(report_file) as writer:
        for network in nn_detailed_stats:
            nn_detailed_stats[network].to_excel(writer, network+"perf")
        for network in nn_energy_stats:
            nn_energy_stats[network].to_excel(writer, network+"energy")

        summary_df.to_excel(writer, 'summary')
    print("**************************************************************************************************************")
    for stat in inference_stats:
        print(stat)
    print("**************************************************************************************************************")
     
def main():
    parser = argparse.ArgumentParser(description="Main program for mapping video analytics pipelines mapping onto PIM hardware")
    parser.add_argument("--pipeline", type=str, default="detect_track")
    parser.add_argument("--pim_accelerator", type=str, default="crossbar")
    parser.add_argument("--accelerator_config", type=str, default="crossbar1")
    parser.add_argument("--accelerator_dataflow", type=str, default="weight_stationary")
    parser.add_argument("--report_prefix", type=str, default="detect_track")
    parser.add_argument("--report_dir", type=str, default="report_dir")
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()