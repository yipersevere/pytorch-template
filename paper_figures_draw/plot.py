import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams["figure.figsize"] = (11,5)

import numpy as np
import pandas as pd
import os
from cycler import cycler


from plot_hepler import mobileNet_cifar100, mobileNet_cifar10,\
                        resetnet50_cifar10, resetnet50_cifar100,\
                        resetnet152_cifar10, resetnet152_cifar100,\
                        inceptionv3_cifar10, inceptionv3_cifar100,\
                        pytorch_cifar10, pytorch_cifar10_loss,\
                        pytorch_ResNet, get_loss_data


def plot_error_fig(errors, layer_index, strDict):
    fig, ax = plt.subplots(1, sharex=True)
    colormap = plt.cm.tab20
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(errors))])
    
    last = len(errors)-1
    elastic_last = len(errors)-2

    for k in range(len(errors)):
        # Plots
        x = np.arange(len(errors[k])) + 1
        if k == elastic_last:
            c_label = strDict["elastic_final_layer_label"]
        elif k == last:
            c_label = strDict["original_layer_label"]
        # elif k == (last+1):
        #     c_label = strDict["note_comment"]
        else:
            c_label = strDict["elastic_intermediate_layer_label"] + str(layer_index[k])
        ax.plot(x, errors[k], label=c_label)
        # Legends
        y = k
        x = len(errors)
        # ax.text(x, y, "%d" % k)
    ax.set_ylabel(strDict["y_label"])
    ax.set_xlabel(strDict["x_label"])
    ax.set_title(strDict["fig_title"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = fig_size

    plt.tight_layout()

    plt.savefig(strDict["save_file_name"], bbox_inches="tight")
    fig.savefig(strDict["save_png_file_name"], bbox_inches="tight")
    plt.close("all")

def add_evaluation_columns_df(data, elastic_list, origin_list, model_name, criteria):
    """
    add new row to "data" dataframe
    """
    row = dict()
    elastic_layer_num = elastic_list.shape[1]
    for i in range(elastic_layer_num):
        col_str = model_name + "_elastic" + "_layer_" + str(i)
        if col_str not in data.columns:
            data[col_str] = []
        
        if criteria == "error": #get accuracy instead of error
            row[col_str] = min(elastic_list.iloc[:,i])
        elif criteria == "f1_score":
            row[col_str] = max(elastic_list.iloc[:,i])
    
    origin_col_str = model_name + "_origin"
    
    if origin_col_str not in data.columns:
        data[origin_col_str]= []

    if criteria == "error": #get accuracy instead of error
        row[origin_col_str] = min(origin_list.iloc[:,-1])
    elif criteria == "f1_score":
        row[origin_col_str] = max(origin_list.iloc[:,-1])

    data = data.append([row],ignore_index=True)
    return data

def plot_model_accuracy_on_CIFAR(dataframe, save_path, data):
    # red dashes, blue squares and green triangles
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.set_prop_cycle(cycler('color', ['yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','gold']))

    for index, row in dataframe.iterrows():
        if "Elastic" in row["model"]:
            ax.semilogx(row["total floating point operations"], row["error"], '*', label=row["model"])
        else:
            ax.semilogx(row["total floating point operations"], row["error"],  markevery=2, drawstyle='steps', marker='o', markersize=3,linewidth=1, label=row["model"])
    plt.xlabel('FLOPs')
    plt.ylabel('Error (%)')

    plt.title("Classification on " + data)
    plt.legend(loc='upper right', prop={'size':6.4})
    # plt.tight_layout(pad=4)
    ax.grid(linestyle='--', linewidth=0.5)
    fig.savefig(save_path+ os.sep +'flops-' + data + '-accuracy-all-models-no-imagenet.pdf')
    fig.savefig(save_path+ os.sep +'flops-' + data + '-accuracy-all-models-no-imagenet.png')


def bar_plot_model_accuracy_on_CIFAR(dataframe, save_path, data):
    '''
    plot bar chart on model accuracy
    '''



def keras_DenseNet121_loss(model="DenseNet121", data="CIFAR10"):
    return 0


def plot_loss_fig(errors, layer_index, strDict):
    fig, ax = plt.subplots(1, sharex=True)
    colormap = plt.cm.tab20
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(errors))])

    # red dashes, blue squares and green triangles
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.set_prop_cycle(cycler('color', ['red','red','green','green']))

    
    last = len(errors)-1
    elastic_last = len(errors)-2

    for k in range(len(errors)):
        # Plots
        x = np.arange(len(errors[k])) + 1
        x = [i + 20 for i in x]
        # if k == elastic_last:
        #     c_label = strDict["elastic_final_layer_label"]
        # elif k == last:
        #     c_label = strDict["original_layer_label"]
        # # elif k == (last+1):
        # #     c_label = strDict["note_comment"]
        if k == 0:
            c_label = "validation loss of Elastic-DenseNet-121 final output"
        elif k == 1:
            c_label = "validation loss of DenseNet-121"
        elif k == 2:
            c_label = "training loss of Elastic-DenseNet-121 final output"
        elif k == 3:
            c_label = "training loss of DenseNet-121"

        else:
            c_label = strDict["elastic_intermediate_layer_label"] + str(layer_index[k])
        if k %2 ==0:
            ax.plot(x, errors[k], label=c_label, linewidth=1)
        else:
            ax.plot(x, errors[k], '--', label=c_label, linewidth=1)
        # Legends
        y = k
        x = len(errors)
        # ax.text(x, y, "%d" % k)
    ax.set_ylabel("loss")
    ax.set_xlabel(strDict["x_label"])
    # ax.set_title(strDict["fig_title"])
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax.grid(linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', prop={'size':6.4})
    
    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = fig_size

    plt.tight_layout()

    plt.savefig("/media/yi/harddrive/codes/Keras_Elastic/local_elasticnn/trainModel_withCIFAR/elastic/plot/keras_CIFAR_10_train_loss_Elastic&Original_DenseNet121_01_01.pdf", bbox_inches="tight")
    fig.savefig("/media/yi/harddrive/codes/Keras_Elastic/local_elasticnn/trainModel_withCIFAR/elastic/plot/keras_CIFAR_10_train_loss_Elastic&Original_DenseNet121_01_01.png", bbox_inches="tight")
    plt.close("all")



def plot_model_loss_on_CIFAR(dataframe, save_path, data):
    # red dashes, blue squares and green triangles
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.set_prop_cycle(cycler('color', ['yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','gold']))

    for index, row in dataframe.iterrows():
        if "Elastic" in row["model"]:
            ax.semilogx(row["total floating point operations"], row["error"], '*', label=row["model"])
        else:
            ax.semilogx(row["total floating point operations"], row["error"],  markevery=2, drawstyle='steps', marker='o', markersize=3,linewidth=1, label=row["model"])
    plt.xlabel('FLOPs')
    plt.ylabel('Error (%)')

    plt.title("Classification on " + data)
    plt.legend(loc='upper right', prop={'size':6.4})
    # plt.tight_layout(pad=4)
    ax.grid(linestyle='--', linewidth=0.5)
    fig.savefig(save_path+ os.sep +'flops-' + data + '-accuracy-all-models-no-imagenet.pdf')
    fig.savefig(save_path+ os.sep +'flops-' + data + '-accuracy-all-models-no-imagenet.png')



def plot_loss():

    losses = []

    loss_origin, loss_elastic, layer_plot_index, captionStrDict = get_loss_data(model="DenseNet121", data="CIFAR10")
    # # loss_origin, loss_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet50", data="CIFAR100")
    
    # # for i in layer_plot_index:
    # #     losses.append(list(loss_elastic.iloc[:, i]))
    
    # # losses.append(list(loss_elastic["val_intermediate_pool2_pool_loss"]))
    # # losses.append(list(loss_elastic["val_intermediate_pool3_pool_loss"]))
    # # losses.append(list(loss_elastic["val_intermediate_pool4_pool_loss"]))
    losses.append(list(loss_elastic["val_final_output_loss"])[30:])
    
    # # 这里的val_loss肯定是不能够并列在一起的,因为这里是sum 总和; 数量级都不在一样, 最好是val_final_output_loss(即elastic model的最后一层的classifier和原始的classifier做对比)

    # # losses.append(list(loss_elastic["val_loss"]))
    
    losses.append(list(loss_origin["val_loss"])[30:])

    # add training loss
    losses.append(list(loss_elastic["final_output_loss"])[30:])
    losses.append(list(loss_origin["loss"])[30:])
    
    
    # loss_df = pd.DataFrame({"val_final_output_loss": list(loss_elastic["val_final_output_loss"])[20:],
    #                         "val_loss": list(loss_origin["val_loss"])[20:]
    #                         })

    # plot_model_loss_on_CIFAR(loss_df, "/media/yi/harddrive/codes/Keras_Elastic/local_elasticnn/trainModel_withCIFAR/elastic/plot", "DenseNet 121")

    plot_loss_fig(losses, layer_plot_index, captionStrDict)

    # cifar_10_model_result = pd.read_csv("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot/cifar10_model_result.csv")
    # plot_model_accuracy_on_CIFAR(cifar_10_model_result, "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot", data="CIFAR-10")
    
    
    
    
    return 0

if __name__ == "__main__":
    
    errors = []


    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet18", data="CIFAR100")
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet50", data="CIFAR100")
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet101", data="CIFAR100")

    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet152_cifar100()
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet152_cifar10()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet152_cifar100()

    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet50_cifar100()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet50_cifar10()
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = inceptionv3_cifar10()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = inceptionv3_cifar100()
    

    # error_origin, error_elastic, layer_plot_index, captionStrDict = mobileNet_cifar100()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = mobileNet_cifar10()
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_cifar10()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_cifar10_loss()
    
    # result_df = pd.DataFrame()
    # result_df = add_evaluation_columns_df(result_df, error_elastic, error_origin, "MobileNets_alpha_0.75", "error")
    
    # f1_origin, f1_elastic, f1_layer_plot_index, f1_captionStrDict = mobileNets_alpha_0_75_F1_cifar100()
    # result_df = add_evaluation_columns_df(result_df, f1_elastic, f1_origin, "MobileNets_alpha_0.75", "f1_score")

    # MobileNets_alpha_0_75_FLOPs_ConvLayer = [22579200, 14450688, 28901376, 14450688, 28901376, 14450688, 28901376, 28901376, 28901376, 28901376, 28901376, 14450688, 28901376]
    # MobileNets_alpha_0_75_FLOPs_OutputLayer = [4848, 9696, 9696, 19392, 19392, 38784, 38784, 38784, 38784, 38784, 38784, 77568, 77568, 3800832]
    # MobileNets_alpha_0_75_FLOPs_block_Conv_Output = [22584048, 14460384, 28911072, 14470080, 28920768, 14489472, 28940160, 28940160, 28940160, 28940160, 28940160, 14528256, 28978944, 3800832]
    # MobileNets_alpha_0_75_FLOPs_Cumulative = [22584048, 37044432, 65955504, 80425584, 109346352, 123835824, 152775984, 181716144, 210656304, 239596464, 268536624, 283064880, 312043824, 315844656]




    # # best_acc_df = add_criteria_columns_df(best_acc_df, error_elastic, error_origin, "InceptionV3", "F1 score")


    # result_df.to_json("result_Inception.json")


    # for i in layer_plot_index:
    #     errors.append(list(error_elastic.iloc[:, i]))
    
    # errors.append(list(error_origin.iloc[:,0]))

    # plot_error_fig(errors, layer_plot_index, captionStrDict)

    # cifar_10_model_result = pd.read_csv("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot/cifar10_model_result.csv")
    # plot_model_accuracy_on_CIFAR(cifar_10_model_result, "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot", data="CIFAR-10")

    # cifar_100_model_result = pd.read_csv("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot/cifar100_model_result.csv")
    # plot_model_accuracy_on_CIFAR(cifar_100_model_result, "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot", data="CIFAR-100")



#############################################################################################################################################################
    # plot loss
    plot_loss()

