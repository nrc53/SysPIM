import numpy as np
import math
import sys

# crossbar implementation for various layers
class in_dram_arch:
    def __init__(self, in_dram_config, comp_precsn, comp_mode):
        self.in_dram_config = in_dram_config
        self.dram_banks = in_dram_config['num_banks']
        self.mac_units_per_bank = in_dram_config['mac_units_per_bank']
        # mac_units in each bank can be organized into smaller mac PEs
        self.min_mac_dim = in_dram_config['min_mac_dim'] 
        # check the equation
        self.alu_units = (self.mac_units_per_bank / 4) * np.log(self.mac_units_per_bank/4) * self.dram_banks
        self.mult_units = (self.mac_units_per_bank / 2) * self.dram_banks
 
    def conv(self, input_dims, layer_params):
        kernel_dims = layer_params['kernel']
        padding = layer_params['padding']  
        stride = layer_params['stride']
        output_dim_r = round((input_dims[0] - kernel_dims[0] + padding) / stride) + 1
        output_dim_c = round((input_dims[1] - kernel_dims[1] + padding) / stride) + 1
        operands_per_mac = np.prod(kernel_dims[:-1]) * 2
        # if mac units are 32, we can decompose into 8, 16 smaller units
        if operands_per_mac > self.mac_units_per_bank:
            cycles_per_mac = math.ceil(operands_per_mac / self.mac_units_per_bank)
        elif operands_per_mac <= self.min_mac_dim:
            cycles_per_mac = self.min_mac_dim / self.mac_units_per_bank
        elif operands_per_mac <= (2 * self.min_mac_dim) \
            and (2 * self.min_mac_dim) <= self.mac_units_per_bank:
            cycles_per_mac = (2 * self.min_mac_dim) / self.mac_units_per_bank
        else:
            cycles_per_mac = 1
        
        tot_num_macs = output_dim_r * output_dim_c *kernel_dims[-1]
        tot_cycles = math.ceil(math.ceil(tot_num_macs * cycles_per_mac) / self.dram_banks)

        output_dims = [output_dim_r, output_dim_c, kernel_dims[-1]]

        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)
        
        exec_stats = {'num_banks':self.dram_banks, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read}
        return exec_stats

    def conv_depthwise(self, input_dims, layer_params):
        kernel_dims = layer_params['kernel']
        operands_per_mac = np.prod(kernel_dims[:-1]) * 2
        padding = layer_params['padding']  
        output_dim_r = round((input_dims[0] - kernel_dims[0] + padding) / layer_params['stride']) + 1
        output_dim_c = round((input_dims[1] - kernel_dims[1] + padding) / layer_params['stride']) + 1
        tot_num_macs = output_dim_r * output_dim_c * kernel_dims[-1]
        output_dims = [output_dim_r, output_dim_c, kernel_dims[-1]]
        cycles_per_mac = math.ceil(operands_per_mac / self.mac_units_per_bank)

        tot_cycles = math.ceil((cycles_per_mac * tot_num_macs) / self.dram_banks)
        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)
        
        exec_stats = {'num_banks':self.dram_banks, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read}
        return exec_stats

    def fc(self, input_dims, layer_params):
        input_dims = [1, np.prod(input_dims)]
        kernel_dims = layer_params['kernel']
        num_wt_elems = kernel_dims[0] * kernel_dims[1] 
        operands_per_mac = kernel_dims[0] * 2
        tot_num_macs = kernel_dims[1] * input_dims [0]
        cycles_per_mac = math.ceil(operands_per_mac / self.mac_units_per_bank)
        tot_cycles = math.ceil(cycles_per_mac * (tot_num_macs) * input_dims[0]) / self.dram_banks
        output_dims = [input_dims[0], kernel_dims[1]]
        if layer_params['activation'] == 'ReLu':
            tot_cycles += self.relu(output_dims)

        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        kernel_bytes_read = np.prod(kernel_dims)
        
        exec_stats = {'num_banks':self.dram_banks, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':kernel_bytes_read}
        return exec_stats       

    def relu(self, input_dims):
        tot_num_ops = np.prod(input_dims) #input_dims[0] * input_dims[1] * input_dims[2]
        tot_cycles = math.ceil(tot_num_ops / self.alu_units)
        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(input_dims)
        return tot_cycles
        # exec_stats = {'exec_cycles':tot_cycles, 'output_dims':input_dims, 'input_bytes_read':inp_bytes_read, \
        #     'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}

        # return exec_stats

    def residual_add(self, input_dims):
        tot_num_ops = input_dims[0] * input_dims[1] * input_dims[2]
        tot_cycles = math.ceil(tot_num_ops / self.alu_units)
        inp_bytes_read = np.prod(input_dims) * 2
        output_bytes_written = np.prod(input_dims)
        exec_stats = {'num_banks':self.alu_units, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':input_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def pool(self, input_dims, layer_params):
        num_alus = self.alu_units
        kernel_dims = layer_params['kernel']
        elems_per_kernel = kernel_dims[0] * kernel_dims[1]
        ops_per_kernel = math.floor(elems_per_kernel * np.log(elems_per_kernel))
        num_kernels_r = round((input_dims[0] - kernel_dims[0]) / layer_params['stride'][0]) + 1
        num_kernels_c = round((input_dims[1] - kernel_dims[1]) / layer_params['stride'][1]) + 1
        num_kernels = num_kernels_r * num_kernels_c * input_dims[2] 
        if layer_params['pool_op'] == 'avg':            
            tot_cycles = math.ceil((ops_per_kernel * num_kernels) / self.alu_units)
            tot_div_ops = num_kernels
        if layer_params['pool_op'] == 'max':
            tot_cycles = math.ceil((ops_per_kernel * num_kernels) / self.alu_units)
        output_dims = [num_kernels_r, num_kernels_c, input_dims[2]]
        inp_bytes_read = np.prod(input_dims)
        output_bytes_written = np.prod(output_dims)
        exec_stats = {'num_banks':self.alu_units, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':inp_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def concat(self, input_dims):
        concat_dim = 0
        for dim in input_dims:
            concat_dim += dim[-1]
        output_dims = [input_dims[0][0], input_dims[0][1], concat_dim]
        exec_stats = {'num_banks':self.dram_banks, 'exec_cycles':0, 'input_dims':input_dims, 'output_dims':output_dims, 'input_bytes_read':0, \
            'output_bytes_written':0, 'kernel_bytes_read':0}
        return exec_stats

    def elem_wise_mult(self, input_dims):
        tot_mult_ops = input_dims[0] + input_dims[1] # Specific to the LSTM implementation
        num_alus = self.mult_units
        tot_cycles = math.ceil(tot_mult_ops / num_alus) * 2
        input_bytes_read = tot_mult_ops * 2
        output_bytes_written = input_dims[0]
        exec_stats = {'num_banks':self.mult_units, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':[input_dims[0]], 'input_bytes_read':input_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats

    def elem_wise_add(self, input_dims):
        tot_add_ops = input_dims[0]  # Specific to the LSTM implementation
        tot_cycles = math.ceil(tot_add_ops / self.alu_units)
        input_bytes_read = tot_add_ops * 2
        output_bytes_written = input_dims[0]
        exec_stats = {'num_banks':self.alu_units, 'exec_cycles':tot_cycles, 'input_dims':input_dims, 'output_dims':[input_dims[0]], 'input_bytes_read':input_bytes_read, \
            'output_bytes_written':output_bytes_written, 'kernel_bytes_read':0}
        return exec_stats


        
        