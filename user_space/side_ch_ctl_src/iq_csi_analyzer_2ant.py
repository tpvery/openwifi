#
# openwifi iq analyzer for 2ant (CSI/etc.)
# Xianjun jiao. putaoshu@msn.com; xianjun.jiao@imec.be
# 
# Enable dual antenna mode
# ./side_ch_ctl wh3h11
# 
# Assumed FPGA loopback setting of side_ch
# # iq_len_init=440
# # pre trigger length 2
# ./side_ch_ctl wh11d2
# # trigger condition: tx_intf_iq0_non_zero&(retrans_in_progress==0)
# ./side_ch_ctl wh8d18
# # fpga loopback
# ./side_ch_ctl wh5h4
# #start_idx_ltf = 1+160+32

# # over the air loopback
# ./sdrctl dev sdr0 set reg xpu 1 1
# ./side_ch_ctl wh5h0
# #start_idx_ltf = 1+160+32+70

import os
import sys
import socket
import numpy as np
import matplotlib.pyplot as plt

ltf_fd_ref = np.array([1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,100,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1], dtype='int16')
ltf_fd_fft_ref = np.array([100,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1], dtype='int16')
ltf_td_ref = np.array([31719,-1040,8069,19657,4286,12144,-23372,-7778,19801,10828,201,-27771,4969,11910,-4564,24206,12688,7494,-11613,-26646,16690,14120,-12243,-11460,-7113,-24743,-25847,15240,-570,-18653,18618,2494,-31719,2494,18618,-18653,-570,15240,-25847,-24743,-7113,-11460,-12243,14120,16690,-26646,-11613,7494,12688,24206,-4564,11910,4969,-27771,201,10828,19801,-7778,-23372,12144,4286,19657,8069,-1040], dtype='int16') + \
          1j*np.array([0,-24426,-22565,16808,5661,-17804,-11202,-21553,-5255,827,-23346,-9618,-11882,-3033,32613,-831,-12688,19964,7978,13241,18748,2867,16501,-4426,-30630,-3363,-4162,-15030,10916,23371,21492,19813,0,-19813,-21492,-23371,-10916,15030,4162,3363,30630,4426,-16501,-2867,-18748,-13241,-7978,-19964,12688,831,-32613,3033,11882,9618,23346,-827,5255,21553,11202,17804,-5661,-16808,22565,24426],dtype='int16')

ant0_ltf_corr_peak_phase_store0 = np.zeros(128,)
ant0_ltf_corr_peak_phase_store1 = np.zeros(128,)
ant1_ltf_corr_peak_phase_store0 = np.zeros(128,)
ant1_ltf_corr_peak_phase_store1 = np.zeros(128,)
ant_diff_ltf_corr_peak_phase_store0 = np.zeros(128,)
ant_diff_ltf_corr_peak_phase_store1 = np.zeros(128,)

phase_rotation_for_advanced_sample = np.zeros(65,) + 1j*np.zeros(65,)

def display_iq(iq0_capture, iq1_capture):
    fig_iq_capture = plt.figure(0)
    fig_iq_capture.clf()

    ax_iq0 = fig_iq_capture.add_subplot(211)
    # ax_iq0.set_xlabel("sample")
    ax_iq0.set_ylabel("I/Q")
    ax_iq0.set_title("rx0 I/Q")
    plt.plot(iq0_capture.real, 'b')
    plt.plot(iq0_capture.imag, 'r')
    plt.ylim(-32767, 32767)

    ax_iq1 = fig_iq_capture.add_subplot(212)
    ax_iq1.set_xlabel("sample")
    ax_iq1.set_ylabel("I/Q")
    ax_iq1.set_title("rx1 I/Q")
    plt.plot(iq1_capture.real, 'b')
    plt.plot(iq1_capture.imag, 'r')
    plt.ylim(-32767, 32767)
    fig_iq_capture.canvas.flush_events()

def parse_iq(iq, iq_len):
    # print(len(iq), iq_len)
    num_dma_symbol_per_trans = 1 + iq_len
    num_int16_per_trans = num_dma_symbol_per_trans*4 # 64bit per dma symbol
    num_trans = round(len(iq)/num_int16_per_trans)
    # print(len(iq), iq.dtype, num_trans)
    iq = iq.reshape([num_trans, num_int16_per_trans])
    
    timestamp = iq[:,0] + pow(2,16)*iq[:,1] + pow(2,32)*iq[:,2] + pow(2,48)*iq[:,3]
    iq0_capture = np.int16(iq[:,4::4]) + np.int16(iq[:,5::4])*1j
    iq1_capture = np.int16(iq[:,6::4]) + np.int16(iq[:,7::4])*1j
    # print(num_trans, iq_len, iq0_capture.shape, iq1_capture.shape)

    return timestamp, iq0_capture, iq1_capture

def ltf_td_corr(iq, start_idx_ltf):
    num_sample_ltf = 64
    iq_for_corr = iq[(start_idx_ltf-5):(start_idx_ltf+10+num_sample_ltf)]
    corr_out0 = np.convolve(iq_for_corr,np.conj(np.flip(ltf_td_ref)), 'valid')
    iq_for_corr = iq[(start_idx_ltf+64-5):(start_idx_ltf+64+10+num_sample_ltf)]
    corr_out1 = np.convolve(iq_for_corr,np.conj(np.flip(ltf_td_ref)), 'valid')
    corr_out0_abs2 = np.real(np.multiply(corr_out0, np.conj(corr_out0)))
    corr_out1_abs2 = np.real(np.multiply(corr_out1, np.conj(corr_out1)))
    max_idx_corr_out0 = np.argmax(corr_out0_abs2)
    max_idx_corr_out1 = np.argmax(corr_out1_abs2)
    # print(max_idx_corr_out0, max_idx_corr_out1)
    phase_peak0 = corr_out0[max_idx_corr_out0]
    phase_peak1 = corr_out1[max_idx_corr_out1]
    # phase_peak0 = np.angle(corr_out0[max_idx_corr_out0])
    # phase_peak1 = np.angle(corr_out1[max_idx_corr_out1])
    start_idx_ltf_new = start_idx_ltf + (max_idx_corr_out0-5)
    ltf_td_average2 = iq[(start_idx_ltf_new-3):(start_idx_ltf_new-3+num_sample_ltf)] + iq[(start_idx_ltf_new+64-3):(start_idx_ltf_new+64-3+num_sample_ltf)]

    return corr_out0_abs2, corr_out1_abs2, phase_peak0, phase_peak1, ltf_td_average2

def display_ltf_corr(ant0_corr_out0_abs2, ant0_corr_out1_abs2, ant0_phase_peak0, ant0_phase_peak1, ant1_corr_out0_abs2, ant1_corr_out1_abs2, ant1_phase_peak0, ant1_phase_peak1):
    ant0_ltf_corr_peak_phase_store0[:(128-1)] = ant0_ltf_corr_peak_phase_store0[1:]
    ant0_ltf_corr_peak_phase_store0[(128-1):] = np.angle(ant0_phase_peak0)
    ant0_ltf_corr_peak_phase_store1[:(128-1)] = ant0_ltf_corr_peak_phase_store1[1:]
    ant0_ltf_corr_peak_phase_store1[(128-1):] = np.angle(ant0_phase_peak1)

    ant1_ltf_corr_peak_phase_store0[:(128-1)] = ant1_ltf_corr_peak_phase_store0[1:]
    ant1_ltf_corr_peak_phase_store0[(128-1):] = np.angle(ant1_phase_peak0)
    ant1_ltf_corr_peak_phase_store1[:(128-1)] = ant1_ltf_corr_peak_phase_store1[1:]
    ant1_ltf_corr_peak_phase_store1[(128-1):] = np.angle(ant1_phase_peak1)

    ltf_corr = plt.figure(0)
    ltf_corr.clf()

    ant0_ax_ltf_corr_abs2 = ltf_corr.add_subplot(221)
    ant0_ax_ltf_corr_abs2.set_title("ANT0 LTF TD CORR abs2")
    plt.plot(ant0_corr_out0_abs2, 'b')
    plt.plot(ant0_corr_out1_abs2, 'r+')
    ant0_ax_ltf_corr_phase = ltf_corr.add_subplot(222)
    ant0_ax_ltf_corr_phase.set_title("ANT0 LTF TD CORR phase")
    plt.plot(ant0_ltf_corr_peak_phase_store0, 'b')
    plt.plot(ant0_ltf_corr_peak_phase_store1, 'r+')

    ant1_ax_ltf_corr_abs2 = ltf_corr.add_subplot(223)
    ant1_ax_ltf_corr_abs2.set_title("ANT1 LTF TD CORR abs2")
    plt.plot(ant1_corr_out0_abs2, 'b')
    plt.plot(ant1_corr_out1_abs2, 'r+')
    ant1_ax_ltf_corr_phase = ltf_corr.add_subplot(224)
    ant1_ax_ltf_corr_phase.set_title("ANT1 LTF TD CORR phase")
    plt.plot(ant1_ltf_corr_peak_phase_store0, 'b')
    plt.plot(ant1_ltf_corr_peak_phase_store1, 'r+')

    ltf_corr.canvas.flush_events()

    # show ANT0/1 DIFF
    ltf_corr_ant_diff = plt.figure(1)
    ltf_corr_ant_diff.clf()

    ant_diff_ltf_corr_peak_phase_store0[:(128-1)] = ant_diff_ltf_corr_peak_phase_store0[1:]
    ant_diff_ltf_corr_peak_phase_store0[(128-1):] = np.angle(ant0_phase_peak0*np.conj(ant1_phase_peak0))
    ant_diff_ltf_corr_peak_phase_store1[:(128-1)] = ant_diff_ltf_corr_peak_phase_store1[1:]
    ant_diff_ltf_corr_peak_phase_store1[(128-1):] = np.angle(ant0_phase_peak1*np.conj(ant1_phase_peak1))
    
    ant_diff_ax_ltf_corr_phase = ltf_corr_ant_diff.add_subplot(111)
    ant_diff_ax_ltf_corr_phase.set_title("ANT0/1 LTF TD CORR phase DIFF")
    plt.plot(ant_diff_ltf_corr_peak_phase_store0, 'b')
    plt.plot(ant_diff_ltf_corr_peak_phase_store1, 'r+')

    ltf_corr_ant_diff.canvas.flush_events()

def csi_calculation(ltf_td_average2):
    ltf_fft = np.fft.fft(ltf_td_average2, 64)
    ltf_fft = np.multiply(ltf_fft, phase_rotation_for_advanced_sample[0:64])
    csi = np.multiply(ltf_fft, ltf_fd_fft_ref)
    # csi[27:11] = 0
    csi[0] = (csi[1] + csi[63])/2
    # normalize the power
    # print(csi.shape)
    csi_power = np.sum(np.real(np.multiply(np.conj(csi),csi)))
    if csi_power > 0:
      csi = csi/np.sqrt(csi_power)
    # csi_for_show = np.concatenate((csi[38:64], csi[0:27]))
    csi_for_show = np.concatenate((csi[32:64], csi[0:32]))
    cir = np.fft.ifft(csi)
    cir_for_show = np.concatenate((cir[32:64], cir[0:32]))

    # # NOT USED CSI PROCESSING METHOD
    # csi_proc = np.concatenate((csi[0:32], np.zeros(3*64,), csi[32:64]))
    # csi_proc_for_show = np.concatenate((csi_proc[128:256], csi_proc[0:128]))
    # cir_proc = np.fft.ifft(csi_proc)
    # cir_proc_for_show = np.concatenate((cir_proc[128:256], cir_proc[0:128]))

    # cir_proc = np.concatenate((cir[0:16], np.zeros(32,), cir[48:64]))
    # cir_proc_for_show = np.concatenate((cir_proc[32:64], cir_proc[0:32]))
    # csi_proc = np.fft.fft(cir_proc)
    # csi_proc_for_show = np.concatenate((csi_proc[32:64], csi_proc[0:32]))

    # csi_proc = np.concatenate((csi_for_show[6], csi_for_show[6], csi_for_show[6:59], csi_for_show[58], csi_for_show[58]))
    # # END OF NOT USED CSI PROCESSING METHOD
    
    # # moving average 5 for csi processing
    # csi_proc = np.zeros(57,) + 1j*np.zeros(57,)
    # csi_proc[0] = csi_for_show[6]
    # csi_proc[1] = csi_for_show[6]
    # csi_proc[2:55] = csi_for_show[6:59]
    # csi_proc[55] = csi_for_show[58]
    # csi_proc[56] = csi_for_show[58]
    # moving_average_coef = np.ones(5,)

    # moving average 7 for csi processing
    csi_proc = np.zeros(59,) + 1j*np.zeros(59,)
    csi_proc[0] = csi_for_show[6]
    csi_proc[1] = csi_for_show[6]
    csi_proc[2] = csi_for_show[6]
    csi_proc[3:56] = csi_for_show[6:59]
    csi_proc[56] = csi_for_show[58]
    csi_proc[57] = csi_for_show[58]
    csi_proc[58] = csi_for_show[58]
    moving_average_coef = np.ones(7,)

    csi_proc = np.convolve(csi_proc, moving_average_coef, 'valid')

    csi_proc = np.concatenate((csi_proc[26:], np.zeros(11,), csi_proc[0:26]))
    csi_proc_for_show = np.concatenate((csi_proc[32:64], csi_proc[0:32]))
    cir_proc = np.fft.ifft(csi_proc)
    cir_proc_for_show = np.concatenate((cir_proc[32:64], cir_proc[0:32]))

    return csi_for_show, cir_for_show, csi_proc_for_show, cir_proc_for_show

def display_csi(csi, cir, csi_proc, cir_proc, figure_idx, title_prefix):
    csi_fig = plt.figure(figure_idx)
    csi_fig.clf()

    csi_abs = csi_fig.add_subplot(321)
    csi_abs.set_title(title_prefix+" CSI ABS")
    plt.plot(np.abs(csi[6:59]), 'b+-')
    plt.axis([0, 52, 0, 0.3])
    plt.grid()
    cir_abs = csi_fig.add_subplot(322)
    cir_abs.set_title(title_prefix+" CIR ABS")
    plt.plot(np.abs(cir), 'b+-')
    plt.axis([0, 63, 0, 0.12])
    plt.grid()
    csi_abs_proc = csi_fig.add_subplot(323)
    csi_abs_proc.set_title(title_prefix+" CSI ABS proc")
    plt.plot(np.abs(csi_proc[6:59]), 'b+-')
    plt.axis([0, 52, 0, 1])
    plt.grid()
    cir_abs_proc = csi_fig.add_subplot(324)
    cir_abs_proc.set_title(title_prefix+" CIR ABS proc")
    plt.plot(np.abs(cir_proc), 'b+-')
    plt.axis([0, 63, 0, 0.7])
    plt.grid()
    csi_phase_proc = csi_fig.add_subplot(325)
    csi_phase_proc.set_title(title_prefix+" CSI phase proc")
    csi_proc_phase = np.angle(csi_proc)
    csi_proc_phase[6:59] = np.unwrap(csi_proc_phase[6:59])
    csi_proc_phase[6:59] = csi_proc_phase[6:59] - csi_proc_phase[32]
    plt.plot(csi_proc_phase[6:59], 'b+-')
    plt.axis([0, 52, -3.15, 3.15])
    plt.grid()
    cir_phase_proc = csi_fig.add_subplot(326)
    cir_phase_proc.set_title(title_prefix+" CIR phase proc")
    plt.plot(np.angle(cir_proc), 'b+-')
    plt.axis([0, 63, -3.15, 3.15])
    plt.grid()

    csi_fig.canvas.flush_events()

def display_csi_waterfall_ant0(csi_proc, cir_proc, figure_idx):
    display_csi_waterfall_ant0.csi_abs_waterfall_store = np.roll(display_csi_waterfall_ant0.csi_abs_waterfall_store, 1, axis=0)
    abs_csi_proc_tmp = np.abs(csi_proc[6:59])
    display_csi_waterfall_ant0.csi_abs_waterfall_store[0,:] = abs_csi_proc_tmp

    display_csi_waterfall_ant0.csi_phase_waterfall_store = np.roll(display_csi_waterfall_ant0.csi_phase_waterfall_store, 1, axis=0)
    csi_proc_phase = np.angle(csi_proc)
    csi_proc_phase[6:59] = np.unwrap(csi_proc_phase[6:59])
    csi_proc_phase[6:59] = csi_proc_phase[6:59] - csi_proc_phase[32]
    display_csi_waterfall_ant0.csi_phase_waterfall_store[0,:] = csi_proc_phase[6:59]

    waterfall_fig = plt.figure(figure_idx)
    waterfall_fig.clf()

    ax_abs_csi = waterfall_fig.add_subplot(221)
    ax_abs_csi.set_title('ANT0 CSI amplitude')
    ax_abs_csi.set_xlabel("subcarrier idx")
    plt.plot(abs_csi_proc_tmp, 'b+-')

    ax_abs_cir = waterfall_fig.add_subplot(222)
    ax_abs_cir.set_title('ANT0 CIR amplitude')
    ax_abs_cir.set_xlabel("time")
    plt.plot(np.abs(cir_proc[29:45]), 'b+-')

    ax_abs_csi_waterfall = waterfall_fig.add_subplot(223)
    ax_abs_csi_waterfall.set_title('ANT0 CSI amplitude')
    ax_abs_csi_waterfall.set_xlabel("subcarrier idx")
    ax_abs_csi_waterfall.set_ylabel("time")
    ax_abs_csi_waterfall_shw = ax_abs_csi_waterfall.imshow(display_csi_waterfall_ant0.csi_abs_waterfall_store)      
    plt.colorbar(ax_abs_csi_waterfall_shw)

    ax_phase_csi_waterfall = waterfall_fig.add_subplot(224)
    ax_phase_csi_waterfall.set_title('ANT0 CSI phase')
    ax_phase_csi_waterfall.set_xlabel("subcarrier idx")
    ax_phase_csi_waterfall.set_ylabel("time")
    ax_phase_csi_waterfall_shw = ax_phase_csi_waterfall.imshow(display_csi_waterfall_ant0.csi_phase_waterfall_store)
    plt.colorbar(ax_phase_csi_waterfall_shw)

    waterfall_fig.canvas.flush_events()

def display_csi_waterfall_ant1(csi_proc, cir_proc, figure_idx):
    display_csi_waterfall_ant1.csi_abs_waterfall_store = np.roll(display_csi_waterfall_ant1.csi_abs_waterfall_store, 1, axis=0)
    abs_csi_proc_tmp = np.abs(csi_proc[6:59])
    display_csi_waterfall_ant1.csi_abs_waterfall_store[0,:] = abs_csi_proc_tmp

    display_csi_waterfall_ant1.csi_phase_waterfall_store = np.roll(display_csi_waterfall_ant1.csi_phase_waterfall_store, 1, axis=0)
    csi_proc_phase = np.angle(csi_proc)
    csi_proc_phase[6:59] = np.unwrap(csi_proc_phase[6:59])
    csi_proc_phase[6:59] = csi_proc_phase[6:59] - csi_proc_phase[32]
    display_csi_waterfall_ant1.csi_phase_waterfall_store[0,:] = csi_proc_phase[6:59]

    waterfall_fig = plt.figure(figure_idx)
    waterfall_fig.clf()

    ax_abs_csi = waterfall_fig.add_subplot(221)
    ax_abs_csi.set_title('ANT1 CSI amplitude')
    ax_abs_csi.set_xlabel("subcarrier idx")
    plt.plot(abs_csi_proc_tmp, 'b+-')

    ax_abs_cir = waterfall_fig.add_subplot(222)
    ax_abs_cir.set_title('ANT1 CIR amplitude')
    ax_abs_cir.set_xlabel("time")
    plt.plot(np.abs(cir_proc[29:45]), 'b+-')

    ax_abs_csi_waterfall = waterfall_fig.add_subplot(223)
    ax_abs_csi_waterfall.set_title('ANT1 CSI amplitude')
    ax_abs_csi_waterfall.set_xlabel("subcarrier idx")
    ax_abs_csi_waterfall.set_ylabel("time")
    ax_abs_csi_waterfall_shw = ax_abs_csi_waterfall.imshow(display_csi_waterfall_ant1.csi_abs_waterfall_store)
    plt.colorbar(ax_abs_csi_waterfall_shw)

    ax_phase_csi_waterfall = waterfall_fig.add_subplot(224)
    ax_phase_csi_waterfall.set_title('ANT1 CSI phase')
    ax_phase_csi_waterfall.set_xlabel("subcarrier idx")
    ax_phase_csi_waterfall.set_ylabel("time")
    ax_phase_csi_waterfall_shw = ax_phase_csi_waterfall.imshow(display_csi_waterfall_ant1.csi_phase_waterfall_store)
    plt.colorbar(ax_phase_csi_waterfall_shw)

    waterfall_fig.canvas.flush_events()

UDP_IP = "192.168.10.1" #Local IP to listen
UDP_PORT = 4000         #Local port to listen

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 464) # for low latency. 464 is the minimum udp length in our case (CSI only)

# align with side_ch_control.v and all related user space, remote files
MAX_NUM_DMA_SYMBOL = 8192

if len(sys.argv)<2:
    print("Assume iq_len = 440! (Max UDP 65507 bytes; (65507/8)-1 = 8187)")
    iq_len = 440
else:
    iq_len = int(sys.argv[1])
    print(iq_len)
    # print(type(num_eq))

if iq_len>8187:
    iq_len = 8187
    print('Limit iq_len to 8187! (Max UDP 65507 bytes; (65507/8)-1 = 8187)')

num_dma_symbol_per_trans = 1 + iq_len
num_byte_per_trans = 8*num_dma_symbol_per_trans

# if os.path.exists("iq_2ant.txt"):
#     os.remove("iq_2ant.txt")
# iq_fd=open('iq_2ant.txt','a')

plt.ion()

# # detect the loopback mode: FPGA or analog
# if it is fpga loopback, the values will be pre-known
# data, addr = sock.recvfrom(MAX_NUM_DMA_SYMBOL*8) # buffer size
data, addr = sock.recvfrom(num_byte_per_trans)
test_residual = len(data)%num_byte_per_trans
if (test_residual != 0):
    print("Abnormal length")
iq = np.frombuffer(data, dtype='uint16')
timestamp, iq0_capture, iq1_capture = parse_iq(iq, iq_len)
stf_ref = np.array([0, 2943, -8477, -862, 9136, 5888, 9136, -862, -8477], dtype='int16') + 1j*np.array([0, 2943, 150, -5026, -811, 0, -811, -5026, 150], dtype='int16')
if np.array_equal(iq0_capture[0,0:9], stf_ref):
  print('FPGA loopback')
  start_idx_ltf = 1+160+32
else:
  print('Analog loopback')
  start_idx_ltf = 1+160+32+70

phase_rotation_for_advanced_sample = np.exp(1j*np.linspace(0, 3*2*np.pi, 65))
display_csi_waterfall_ant0.csi_abs_waterfall_store = np.zeros((64, 53))
display_csi_waterfall_ant0.csi_phase_waterfall_store = np.zeros((64, 53))
display_csi_waterfall_ant1.csi_abs_waterfall_store = np.zeros((64, 53))
display_csi_waterfall_ant1.csi_phase_waterfall_store = np.zeros((64, 53))

while True:
    try:
        # data, addr = sock.recvfrom(MAX_NUM_DMA_SYMBOL*8) # buffer size
        data, addr = sock.recvfrom(num_byte_per_trans) 
        # print(addr)
        test_residual = len(data)%num_byte_per_trans
        # print(len(data)/8, num_dma_symbol_per_trans, test_residual)
        if (test_residual != 0):
            print("Abnormal length")

        iq = np.frombuffer(data, dtype='uint16')
        # np.savetxt(iq_fd, iq)
        # print(iq.shape)

        timestamp, iq0_capture, iq1_capture = parse_iq(iq, iq_len)
        # print(timestamp, max(iq0_capture.real), max(iq1_capture.real))
        ant0_corr_out0_abs2, ant0_corr_out1_abs2, ant0_phase_peak0, ant0_phase_peak1, ant0_ltf_td_average2 = ltf_td_corr(iq0_capture[0,:], start_idx_ltf)
        
        # ant1_corr_out0_abs2, ant1_corr_out1_abs2, ant1_phase_peak0, ant1_phase_peak1, ant1_ltf_td_average2 = ltf_td_corr(iq1_capture[0,:], start_idx_ltf)
        # display_ltf_corr(ant0_corr_out0_abs2, ant0_corr_out1_abs2, ant0_phase_peak0, ant0_phase_peak1, ant1_corr_out0_abs2, ant1_corr_out1_abs2, ant1_phase_peak0, ant1_phase_peak1)
        
        ant0_csi, ant0_cir, ant0_csi_proc, ant0_cir_proc = csi_calculation(ant0_ltf_td_average2)
        # display_csi(ant0_csi, ant0_cir, ant0_csi_proc, ant0_cir_proc, 2, 'ANT0')
        
        # ant1_csi, ant1_cir, ant1_csi_proc, ant1_cir_proc = csi_calculation(ant1_ltf_td_average2)
        # display_csi(ant1_csi, ant1_cir, ant1_csi_proc, ant1_cir_proc, 3, 'ANT1')
        
        display_csi_waterfall_ant0(ant0_csi_proc, ant0_cir_proc, 4)
        # display_csi_waterfall_ant1(ant1_csi_proc, ant1_cir_proc, 5)

    except KeyboardInterrupt:
        print('User quit')
        break

print('close()')
# iq_fd.close()
sock.close()
