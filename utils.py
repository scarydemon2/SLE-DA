import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_process.constant import event_type,all_slots
def show_attention_heatmap(slot2token_score,token2slot_score,event2slot_score,event_pred,event_gold_label,argument_pred,argument_gold,tokens):
    for i in range(len(slot2token_score)):#对于batch每个instance
        if (event_pred[i]==event_gold_label[i]).all() and event_gold_label[i][0]!=1:
            fig=plt.figure(figsize=(72, 18))
            plt.legend(prop={"size": 50, "weight": "black"})
            heat_map1=fig.add_subplot(141)
            event_type_array=np.array(event_type)
            gold_label_array = np.array(event_gold_label[i])
            mask=(gold_label_array==1)
            event=event_type_array[mask]
            print(event)
            score=slot2token_score[i]
            score_df=pd.DataFrame(score,index=all_slots ,columns=tokens[i])
            map=sns.heatmap(score_df)
            heat_map1.plot()

            heat_map2=fig.add_subplot(142)
            score=event2slot_score[i]
            score_df=pd.DataFrame(score,index=event_type,columns=all_slots)
            map2=sns.heatmap(score_df)
            heat_map2.plot()

            heat_map3 = fig.add_subplot(143)
            score = event2slot_score[i]@slot2token_score[i]
            score_df = pd.DataFrame(score, index=event_type, columns=tokens[i])
            map3 = sns.heatmap(score_df)
            heat_map3.plot()

            # heat_map4 = fig.add_subplot(144)
            # score = event2seq_score[i]
            # score_df = pd.DataFrame(score, index=event_type, columns=tokens[i])
            # map4 = sns.heatmap(score_df)
            # heat_map4.plot()
            plt.show()
