import math
import pandas as pd
import numpy as np

class crossbar_power:
    def __init__(self, cb_config):
        self.cb_config = cb_config
        adc_power = (self.cb_config['num_adc'] / self.cb_config['power']['num_adc']) * self.cb_config['power']['adc']
        dac_power = (self.cb_config['row_activations'] / self.cb_config['power']['num_dac']) * self.cb_config['power']['dac']
        sh_power = (self.cb_config['array_dims'][1] / self.cb_config['power']['array_cols']) * self.cb_config['power']['s_h'] 
        sa_power = (self.cb_config['array_dims'][1] / self.cb_config['power']['array_cols']) * self.cb_config['power']['s_a']
        rram_power = (np.prod(self.cb_config['array_dims']) / np.prod(self.cb_config['power']['rram_dims'])) * self.cb_config['power']['rram_array']
        self.array_power = adc_power + dac_power + sh_power + sa_power +rram_power
        self.static_weights = self.cb_config['static_weights']

    #exec_stats is a list of dicts, each dict has execution stats for a single layer
    def calc_energy(self, exec_stats):
        tot_exec_cycles = 0
        tot_compute_energy = 0
        tot_input_read_energy = 0
        tot_kernel_read_energy = 0
        tot_output_write_energy = 0
        tot_mem_energy = 0
        layer_name_list = []
        compute_energy_list = []
        input_read_energy_list = []
        kernel_read_energy_list = []
        output_write_energy_list = []
        #self.array_power = cb_power(self.cb_config)

        for layer_stats in exec_stats: 
            num_cb = layer_stats['num_cb']
            exec_cycles = layer_stats['exec_cycles']
            tot_cb_power = self.array_power * num_cb
            input_mem_accesses = math.ceil(layer_stats['input_bytes_read'] / self.cb_config['power']['sram_access_bytes'])
            kernel_mem_accesses = math.ceil(layer_stats['kernel_bytes_read'])
            output_mem_accesses = math.ceil(layer_stats['output_bytes_written'] / self.cb_config['power']['sram_access_bytes'])
            compute_energy = tot_cb_power * exec_cycles * float(self.cb_config['mac_time'])
            input_read_energy = input_mem_accesses * float(self.cb_config['power']['sram_access'])
            if self.static_weights:
                kernel_read_energy = 0
            else:
                kernel_read_energy = kernel_mem_accesses * float(self.cb_config['power']['dram_energy_bit']) * 8
            output_write_energy = output_mem_accesses * float(self.cb_config['power']['sram_access'])
            mem_energy = input_read_energy + kernel_read_energy + output_write_energy
            compute_energy_list.append(compute_energy)
            input_read_energy_list.append(input_read_energy)
            kernel_read_energy_list.append(kernel_read_energy)
            output_write_energy_list.append(output_write_energy)
            layer_name_list.append(layer_stats['name'])
            tot_exec_cycles += exec_cycles
            tot_compute_energy += compute_energy
            tot_input_read_energy += input_read_energy
            tot_kernel_read_energy += kernel_read_energy
            tot_output_write_energy += output_write_energy

        bus_energy = tot_exec_cycles * float(self.cb_config['mac_time']) * float(self.cb_config['power']['bus'])
        layer_name_list.append("Total")
        compute_energy_list.append(tot_compute_energy)
        input_read_energy_list.append(tot_input_read_energy)
        kernel_read_energy_list.append(tot_kernel_read_energy)
        output_write_energy_list.append(tot_output_write_energy)
        bus_energy_list = [bus_energy] * len(compute_energy_list)

        energy_stats_dict = {}
        energy_stats_dict['name'] = layer_name_list
        energy_stats_dict['compute_energy'] = compute_energy_list
        energy_stats_dict['input_read_energy'] = input_read_energy_list
        energy_stats_dict['kernel_read_energy'] = kernel_read_energy_list
        energy_stats_dict['output_write_energy'] = output_write_energy_list
        energy_stats_dict['bus_energy'] = bus_energy_list

        energy_stats_df = pd.DataFrame(energy_stats_dict)
        
        return energy_stats_df