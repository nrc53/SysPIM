import math
import pandas as pd
import numpy as np

class in_cache_power:
    def __init__(self, in_cache_config):
        self.in_cache_config = in_cache_config
        self.mat_tile_energy = (np.prod(self.in_cache_config['comp_tile_size']) / np.prod(self.in_cache_config['power']['mat_tile'])) \
            * self.in_cache_config['power']['mat_tile_energy']
        self.add_energy = float(self.in_cache_config['power']['add_energy'])
        self.mult_energy = float(self.in_cache_config['power']['mult_energy'])
        self.sram_energy_bit = float(self.in_cache_config['power']['sram_energy_bit'])
        self.dram_energy_bit = float(self.in_cache_config['power']['dram_energy_bit'])
        self.alu_energy_PE = 8 * self.add_energy
        self.mult_energy_PE = self.mult_energy

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
        #print(exec_stats)
        dram_bytes = 0
        sram_bytes = 0
        for layer_stats in exec_stats: 
            print(layer_stats)
            if 'conv' in layer_stats['name'] or 'fc' in layer_stats['name']:  #op is tiled multiply
                tot_mult_tiles = layer_stats['mm_tiles']
                compute_energy = self.mat_tile_energy * tot_mult_tiles
                # Add logic for activation layer ReLu
                if layer_stats['activation'] == "ReLu":
                    alu_ops = math.ceil(np.prod(layer_stats['output_dims'])/ 8)
                    compute_energy += self.alu_energy_PE * alu_ops
            elif 'pool' in layer_stats['name']:  #op is alu/add
                tot_alu_ops = layer_stats['mm_tiles']
                compute_energy = self.alu_energy_PE * tot_alu_ops
            elif 'add' in layer_stats['name']:  #op is alu/add
                tot_alu_ops = layer_stats['mm_tiles']
                compute_energy = self.alu_energy_PE * tot_alu_ops
            elif 'mult' in layer_stats['name']: #op is elem-wise mult
                tot_mult_ops = layer_stats['mm_tiles']
                compute_energy = self.mult_energy_PE * tot_mult_ops
            else:
                compute_energy = 0

            # Calculate mem access energy
            if layer_stats['input_bytes_dram']:
                input_read_energy = layer_stats['input_bytes_read'] * self.dram_energy_bit * 8
                dram_bytes += layer_stats['input_bytes_read']
            else:
                input_read_energy = layer_stats['input_bytes_read'] * self.sram_energy_bit * 8
                sram_bytes += layer_stats['input_bytes_read']
            if layer_stats['output_bytes_dram']:
                output_write_energy = layer_stats['output_bytes_written'] * self.dram_energy_bit * 8
                dram_bytes += layer_stats['output_bytes_written']
            else:
                output_write_energy = layer_stats['output_bytes_written'] * self.sram_energy_bit * 8
                sram_bytes += layer_stats['output_bytes_written']
                
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
        print("bytes read/wrote to SRAM/DRAM")
        print(dram_bytes)
        print(sram_bytes)
        print("**************************************************")
        
        return energy_stats_df