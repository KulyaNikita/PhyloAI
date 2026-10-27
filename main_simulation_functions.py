#!/usr/bin/env python3

"""

(c) Nikita Kulikov, Iker Irisarri, Christoph Mayer; LIB 2024

Support functions for the alignment simulations with PolyMoSim

"""
import numpy as np

base_models_choice_aa = [
    "JTT", "LG", "WAG_OLD", "WAG", "WAG_STAR", "DAY", "Qplant",
    "Qpfam", "Qmammal", "Qinsect", "Qbird", "Qlg", "Qyeast", "VT",
    "PMB", "JTTDCMut", "DCMut", "Blosum62",
]

def make_model_file(
    file_name,
    model,
    alpha,
    pinv,
    tstv,
    basefreq_1,
    basefreq_2,
    GTR_A_C,
    GTR_A_G,
    GTR_A_T,
    GTR_C_G,
    GTR_C_T,
    GTR_G_T,
    model_name,
):
    with open(file_name, "w") as model_file:
        if model == "JC":
            model_file.write("begin model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            model_file.write("end model\n")
        elif model == "K2P":
            model_file.write("begin model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            string = "tstv: {0}\n".format(tstv)
            model_file.write(string)
            model_file.write("end model\n")
        elif model == "F81":
            model_file.write("begin model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            string = "basefreq: {0}  {1}  {2}  {3}\n".format(basefreq_1, basefreq_2, basefreq_1, basefreq_2)
            model_file.write(string)
            model_file.write("end model\n")
        elif model == "HKY":
            model_file.write("begin model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            string = "tstv: {0}\n".format(tstv)
            model_file.write(string)
            string = "basefreq: {0}  {1}  {2}  {3}\n".format(basefreq_1, basefreq_2, basefreq_1, basefreq_2)
            model_file.write(string)
            model_file.write("end model\n")
        elif model == "F84":
            model_file.write("begin model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            string = "tstv: {0}\n".format(tstv)
            model_file.write(string)
            string = "basefreq: {0}  {1}  {2}  {3}\n".format(basefreq_1, basefreq_2, basefreq_1, basefreq_2)
            model_file.write(string)
            model_file.write("end model\n")
        elif model == "GTR":
            model_file.write("begin model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            string = "basefreq: {0}  {1}  {2}  {3}\n".format(basefreq_1, basefreq_2, basefreq_1, basefreq_2)
            model_file.write(string)
            string = "rrates: {} {} {} {} {} {}\n".format(GTR_A_C, GTR_A_G, GTR_A_T, GTR_C_G, GTR_C_T, GTR_G_T)
            model_file.write(string)
            model_file.write("end model\n")
        elif model in base_models_choice_aa:
            model_file.write("begin aa-model\n")
            model_file.write("name: {0}\n".format(model_name))
            model_file.write("modeltype: {0}\n ".format(model))
            string = "shape: {0}\n".format(alpha)
            model_file.write(string)
            string = "pinv: {0}\n".format(pinv)
            model_file.write(string)
            model_file.write("end model\n")
            print("Model file written:")

def make_tree_file_model(file_name, topology, B1, B2, B3, B4, B5, B6, seqlen, modelname):
    with open(file_name, "w") as tree_file:
        if topology == 0:
            string = "1.0  {0} {1} ((A:{2},B:{3}):{4},(C:{5},D:{6}):{7})\n".format(
                seqlen, modelname, B1, B2, B3, B4, B5, B6
            )
        if topology == 1:
            string = "1.0  {0} {1} ((A:{2},C:{3}):{4},(B:{5},D:{6}):{7})\n".format(
                seqlen, modelname, B1, B2, B3, B4, B5, B6
            )
        if topology == 2:
            string = "1.0  {0} {1} ((A:{2},D:{3}):{4},(C:{5},B:{6}):{7})\n".format(
                seqlen, modelname, B1, B2, B3, B4, B5, B6
            )
        print(string)
        tree_file.write(string)

def find_frequency(freq_file):
    with open(freq_file, "r") as frequency_file:
        frequency_list = []
        for line in frequency_file:
            column = line.split("\t")
            if len(column[0]) == 4:
                frequency_list.append(np.float32(column[1]))
            else:
                f_e = open("file_error.txt", "a")
                f_e.write(line)
                f_e.write("\n")
                f_e.close()
    return frequency_list
