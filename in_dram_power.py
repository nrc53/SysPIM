import math
import pandas as pd
import numpy as np

class in_dram_power:
    def __init__(self, in_dram_config):
        self.in_dram_config = in_dram_config
        self.mac_power = float(self.in_dram_config['power']['mac_power'])
        self.alu_power = float(self.in_dram_config['power']['alu_power'])
        self.mult_power = float(self.in_dram_config['power']['mult_power'])
        self.dram_energy_bit = float(self.in_dram_config['power']['dram_energy_bit'])
        self.cycle_time = float(self.in_dram_config['cycle_time'])

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
            if 'conv' in layer_stats['name'] or 'fc' in layer_stats['name']:  #op is matmult
                compute_energy = self.mac_power * layer_stats['exec_cycles'] * self.cycle_time * layer_stats['num_banks']
            elif 'pool' in layer_stats['name']:  #op is alu/add
                compute_energy = self.alu_power * layer_stats['exec_cycles'] * self.cycle_time * layer_stats['num_banks']
            elif 'add' in layer_stats['name']:  #op is alu/add
                compute_energy = self.alu_power * layer_stats['exec_cycles'] * self.cycle_time * layer_stats['num_banks']
            elif 'mult' in layer_stats['name']: #op is elem-wise mult
                compute_energy = self.mult_power * layer_stats['exec_cycles'] * self.cycle_time * layer_stats['num_banks']
            else:
                compute_energy = 0

            # Calculate mem access energy

            input_read_energy = layer_stats['input_bytes_read'] * self.dram_energy_bit * 8
            output_write_energy = layer_stats['output_bytes_written'] * self.dram_energy_bit * 8
            kernel_read_energy = layer_stats['kernel_bytes_read'] * self.dram_energy_bit * 8
            compute_energy_list.append(compute_energy)
            input_read_energy_list.append(input_read_energy)
            kernel_read_energy_list.append(kernel_read_energy)
            output_write_energy_list.append(output_write_energy)
            layer_name_list.append(layer_stats['name'])
            tot_compute_energy += compute_energy
            #print("input read energy ", input_read_energy)
            tot_input_read_energy += input_read_energy
            tot_kernel_read_energy += kernel_read_energy
            tot_output_write_energy += output_write_energy

        layer_name_list.append("Total")
        compute_energy_list.append(tot_compute_energy)
        input_read_energy_list.append(tot_input_read_energy)
        kernel_read_energy_list.append(tot_kernel_read_energy)
        output_write_energy_list.append(tot_output_write_energy)
        #bus_energy_list = [bus_energy] * len(compute_energy_list)

        energy_stats_dict = {}
        energy_stats_dict['name'] = layer_name_list
        energy_stats_dict['compute_energy'] = compute_energy_list
        energy_stats_dict['input_read_energy'] = input_read_energy_list
        energy_stats_dict['kernel_read_energy'] = kernel_read_energy_list
        energy_stats_dict['output_write_energy'] = output_write_energy_list
        #energy_stats_dict['bus_energy'] = output_write_energy_list

        energy_stats_df = pd.DataFrame(energy_stats_dict)
        
        return energy_stats_df