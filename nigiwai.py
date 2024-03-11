import json
import string
import os
import pandas as pd
import datetime
from collections import Counter
import itertools
import re
from scipy import spatial
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np
import pyathena
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from googleapiclient.discovery import build
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

# APIキーの設定
apis = pd.read_csv("D:\\code\\FCDL_にぎわい\\API\\apis.csv")
openai.api_key = apis["openai"][0]
API_KEY = apis["googlesearch"][0]

CUSTOM_SEARCH_ENGINE = 'a78a2783d10a44be6'
page_limit = 1
PROMPT_TEXT_COMMON = '''
    この文章の主題を15文字以上30文字未満で要約してください。必ず30文字未満にしてください。
'''

PROMPT_TEXT_FC = '''
    この文章の主題を要約すると次の5個です。それぞれ15文字以上30文字未満で要約してください。
'''

PROMPT_TEXT_OTR = '''
    この文章の主題を要約すると次の3個です。それぞれ15文字以上30文字未満で要約してください。
'''

PROMPT_TEXT_THEME = '''
    この文章の主題の要点を一文で要約してください。
'''
PROMPT_TEXT_DL = '''
    このデータセットの主題を15文字以上50文字未満で要約してください。必ず50文字未満にしてください。
'''

model = SentenceTransformer('stsb-xlm-r-multilingual')

# アウトプットのトークン数。長い文章を返す場合には大きな数字にする。
OUTPUT_TOKENS = 1000

# distanceの数式の変更
MAX_DISTANCE = 3.0
EMPTY_TEXT = ''
SIMILARITY_UPPER_THRESHOLD = 3.00
SIMILARITY_LOWER_THRESHOLD = 2.65
RESCALED_MAX_SIMILARITY_VALUE = 3.0
RESCALED_MIN_SIMILARITY_VALUE = 0.1

distance_fc_list = []
distance_otr_list = []
distance_fc_and_otr_list = []

def getImageUrl(api_key, cse_key, search_word):
    service = build("customsearch", "v1", developerKey=api_key)
    page_limit = 1
    startIndex = 1
    response = []
    img_list = []

    try:
        response.append(service.cse().list(
            q=search_word,     # Search words
            cx=cse_key,        # custom search engine key
            lr='lang_ja',      # Search language
            num=1,            # Number of images obtained by one request (Max 10)
            start=startIndex,
            searchType='image' # search for images
        ).execute())

        startIndex = response[0].get("queries").get("nextPage")[0].get("startIndex")

    except Exception as e:
        print(e)

    for one_res in range(len(response)):
        if int(response[one_res]["searchInformation"]["totalResults"]) > 0:
            for i in range(len(response[one_res]['items'])):
                img_list.append(response[one_res]['items'][i]['link'])
        else:
            img_list.append("https://1.bp.blogspot.com/-d3vDLBoPktU/WvQHWMBRhII/AAAAAAABL6E/Grg-XGzr9jEODAxkRcbqIXu-mFA9gTp3wCLcBGAs/s800/internet_404_page_not_found.png")

    return img_list[0]

# 文章ベクトルを取得する関数の定義
def get_sentence_vector(in_sentence):
    return model.encode(in_sentence, convert_to_tensor=False)

# 二つの文章ベクトルの間の距離を取得する関数の定義
def get_similarity(in_vec_a, in_vec_b):
    return MAX_DISTANCE - spatial.distance.cosine(in_vec_a, in_vec_b)

# インプットのテキストを整形する関数の定義。全角スペースを半角スペースに変換。
def _get_cleaned_text(in_text):
    in_text = in_text.replace('　', ' ')
    return in_text

def _get_empty_text_removed_list(in_list):
    return [s for s in in_list if s != EMPTY_TEXT]

def _get_parsed_result_by_return(in_sentence):
    return _get_empty_text_removed_list(in_sentence.split('\n'))

def _get_numbering_removed_keyword(in_keyword_list):
    this_result = []
    for this_keyword in in_keyword_list:
        this_keyword = re.sub(r'^\d+\. ', '', this_keyword)
        this_result.append(re.sub(r'^\d+\.', '', this_keyword))
    return this_result

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# 結果を取得する関数の定義
def get_results_for_one_text(in_text, prompt):
    response = openai.Completion.create(
        # model="gpt-4",
        # model="gpt-3.5-turbo",
        # model="text-davinci-003",
        model="gpt-3.5-turbo-instruct",
        prompt=_get_cleaned_text(in_text) + prompt,
        temperature=0, #temperature=0.5,
        max_tokens=OUTPUT_TOKENS,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )
    return _get_numbering_removed_keyword(
        _get_parsed_result_by_return(
            response["choices"][0]["text"]
        ))

########################## clustering ##########################
def sentence_vector_preprocessing_before_clustering(fc_df):
    vectors = []
    for i in range(len(fc_df)):
        vectors.append(np.array(fc_df["Vector"][i]))
    norm_vectors = normalize(vectors)
    cosine_sim_vectors = cosine_similarity(vectors)
    return norm_vectors, cosine_sim_vectors


def clustering_metrics_evaluation(input_vectors, fitted_labels, model_title, preprocess_method):
    print("############# Displaying the statistical result of the " + model_title + "with preprocess method of "+ preprocess_method + " #############")
    silhousette = metrics.silhouette_score(input_vectors, fitted_labels)
    print("The Silhousette Score: ", silhousette)

    calinski = metrics.calinski_harabasz_score(input_vectors, fitted_labels)
    print("The Calinski-Harabasz Score: ", calinski)

    Davies = metrics.davies_bouldin_score(input_vectors, fitted_labels)
    print("The Davies-Bouldin Score: ", Davies)

def demonstrating_clustering_result(fitted_labels):
    pass
    # print(fitted_labels)
    
def clustering_Kmeans(fc_df, num_clusters, preprocess_method):
    norm_vectors, cosine_sim_vectors = sentence_vector_preprocessing_before_clustering(fc_df)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    if preprocess_method =='normalization':
        kmeans.fit(norm_vectors)
    elif preprocess_method == 'cosine':
        kmeans.fit(cosine_sim_vectors)
    else:
        fitted_labels = []
        print("Wrong preprocess method!")
    fitted_labels = kmeans.labels_
    fc_df["Cluster"] = fitted_labels
    return kmeans


###################### FCデータセットの取得 ##############################
if False:
    print("Start calc extraction")
    print(datetime.datetime.now())
    opt_df = pd.read_csv("D:\\code\\FCDL_にぎわい\\csv\\にぎわいづくりGSSMrev.csv", encoding='UTF-8')
    fc_question_list = list(opt_df.columns[2:-1])
    fc_sentence_list = []
    fc_vector_list = []
    fc_keyword_list = []
    fc_user_list = []
    for index in range(len(opt_df)):
        for question in fc_question_list:
            fc_sentence = opt_df[question][index]
            user = opt_df["User"][index]
            if type(fc_sentence) == float:
                fc_sentence = ""
            else:
                fc_sentence = fc_sentence.strip()
            if len(fc_sentence) >=3:
                fc_vector_list.append(get_sentence_vector(fc_sentence))

                if len(fc_sentence) < 30:
                    fc_sentence_list.append(fc_sentence)
                    fc_keyword_list.append(fc_sentence)
                    fc_user_list.append(f"【User】{user}")
                else:
                    fc_sentence_list.append("".join(get_results_for_one_text(fc_sentence, PROMPT_TEXT_COMMON)))
                    fc_keyword = get_results_for_one_text(fc_sentence[:OUTPUT_TOKENS], PROMPT_TEXT_COMMON)
                    fc_keyword_list.append(fc_keyword[0].replace("・", "").replace(" ", "").replace("　", "").replace("「", "").replace("」", ""))
                    fc_user_list.append(f"【User】{user}")
    fc_all = pd.DataFrame({
        'Sentence': fc_sentence_list, 
        'Vector': fc_vector_list,
        'Keyword': fc_keyword_list,
        'User': fc_user_list
    })
    print(datetime.datetime.now())
    with open('D:\\code\\FCDL_にぎわい\\pkls\\fc_extraction.pkl', 'wb') as f:
        pickle.dump(fc_all, f)
else:
    with open('D:\\code\\FCDL_にぎわい\\pkls\\fc_extraction.pkl', 'rb') as f:
        fc_all = pickle.load(f)

############## Tokyo Data ################
if True:
    dl_Tokyo = pd.read_csv("D:\\code\\FCDL_にぎわい\\csv\\tokyo_opendata_name.csv", usecols=["datahead", "creator", "corp"])
    dl_Tokyo = dl_Tokyo.drop_duplicates(subset="datahead").head(100).reset_index()
    sentences = []
    creators = []
    corps = []
    for index in range(len(dl_Tokyo)):
        sentence = dl_Tokyo["datahead"][index]
        creator = dl_Tokyo["creator"][index]
        corp = dl_Tokyo["corp"][index]
        if len(sentence) > 50:
            sentence = ''.join(get_results_for_one_text(sentence, PROMPT_TEXT_DL)).strip().strip('「').strip('」').strip('。')
        sentences.append(sentence.strip())
        creators.append(creator)
        corps.append(corp)
    print(len(corps))
    #Compute embeddings
    if True:
        print("Start calc embeddings")
        print(datetime.datetime.now())
        embeddings = model.encode(sentences, convert_to_tensor=True)
        print(datetime.datetime.now())

        dl_tokyo = pd.DataFrame({
        'Sentence': sentences, 
        'Creator' : creators,
        'Corps' : corps,
        'Embedding': embeddings
        })

    # pickle化してファイルに書き込み
    with open('D:\\code\\FCDL_にぎわい\\pkls\\dl__tokyo_extraction.pkl', 'wb') as f:
        pickle.dump(dl_tokyo, f)
    embeddings_dl_tokyo = list(dl_tokyo["Embedding"])
else:
    with open('D:\\code\\FCDL_にぎわい\\pkls\\dl__tokyo_extraction.pkl', 'rb') as f:
        dl_tokyo = pickle.load(f)
    embeddings_dl_tokyo = list(dl_tokyo["Embedding"]) 


############# データジャケットの内容をベクトルとして保存 ####################
if True:
    dls = pd.read_csv("D:\\code\\FCDL_にぎわい\\csv\\datajacket DL.csv", usecols=["ID", "title", "outline", "collecting_cost", "sharing_policy", "type", "variable", "analysis", "outcome", "anticipation", "comments", "wanted"])
    dls["fulltext"] = [f"{ttl}, {ol}, {v}" for ttl, ol, v in zip(dls["title"], dls["outline"], dls["variable"])]

    sentences = []
    dls = dls.head(100)
    for add_s in dls["fulltext"]:
        if len(add_s) > 50:
            add_s = ''.join(get_results_for_one_text(add_s, PROMPT_TEXT_DL)).strip().strip('「').strip('」').strip('。')
        sentences.append(add_s.strip())
    #Compute embeddings
    if True:
        print("Start calc embeddings")
        print(datetime.datetime.now())
        embeddings = model.encode(sentences, convert_to_tensor=True)
        print(datetime.datetime.now())

        dl_all = pd.DataFrame({
        'Sentence': sentences, 
        'Embedding': embeddings
        })

    # pickle化してファイルに書き込み
    with open('D:\\code\\FCDL_にぎわい\\pkls\\dl_extraction.pkl', 'wb') as f:
        pickle.dump(dl_all, f)
    embeddings_dls = list(dl_all["Embedding"])
else:
    with open('D:\\code\\FCDL_にぎわい\\pkls\\dl_extraction.pkl', 'rb') as f:
        dl_all = pickle.load(f)
    embeddings_dls = list(dl_all["Embedding"])     

def clustering_and_create_network(fc_df, embdl, embdltokyo, dlsentence, dltokyo , seg, num_clusters, network_num):
    #k-means法でDLをクラスタリング
    clustering = clustering_Kmeans(fc_df, num_clusters, "normalization")
    fc_emb_df = fc_df[["Sentence", "Cluster"]].groupby(["Cluster"])["Sentence"].apply('。'.join).reset_index()

    emb_list = []
    for i in range(len(clustering.cluster_centers_)):
        emb_list.append(model.encode(fc_emb_df["Sentence"][i], convert_to_tensor=True))
    
    # 入力文と検索対象文のベクトル表現の類似度を計算
    tgt_cluster_list = []
    dl_sentence_list = []
    dl_num_list = []
    dl_dtype_list = []
    dl_creator_list = []
    for emb_i in range(len(emb_list)):
        embedding = emb_list[emb_i]

        # dl
        scores = util.pytorch_cos_sim(embedding, embdl)
        sorted, indices = scores.sort(descending=True)
        for i in range(5):
            predicted_idx = int(indices[0][i]) # スコアが最大のインデックスの取得
            tgt_cluster_list.append(emb_i)
            dl_sentence_list.append(dlsentence["Sentence"][predicted_idx])
            dl_num_list.append(f"DLData,{str(i)}") #　データ名に変更
            dl_dtype_list.append("DLData")  
            dl_creator_list.append("")

        # dlTokyo
        scores = util.pytorch_cos_sim(embedding, embdltokyo)
        sorted, indices = scores.sort(descending=True)
        for i in range(5):
            predicted_idx = int(indices[0][i]) # スコアが最大のインデックスの取得
            tgt_cluster_list.append(emb_i)
            dl_sentence_list.append(dltokyo["Sentence"][predicted_idx])
            dl_num_list.append(f"DLTokyoData,{str(i)}") #　データ名に変更
            dl_dtype_list.append("DLTokyoData")  
            dl_creator_list.append(dltokyo["Creator"][predicted_idx])
        
        # all answer
        fc_flt = fc_df[fc_df["Cluster"] == emb_i].reset_index(drop=True)
        for fc_i in range(len(fc_flt)):
            tgt_cluster_list.append(emb_i)
            dl_sentence_list.append(fc_flt["Keyword"][fc_i])
            dl_dtype_list.append("UserAns")
            dl_num_list.append(f"UserAns,{str(emb_i)}")
            dl_creator_list.append("")
            
    dl_df = pd.DataFrame({
        'Sentence': dl_sentence_list, 
        'User': dl_num_list, 
        'Cluster': tgt_cluster_list, 
        'DataType': dl_dtype_list,
        'Creator' : dl_creator_list
    })
    dl_emb_df = dl_df.copy()

    # 各クラスタからKeyPhraseを取得
    fc_keyword_all = []
    fc_users = []
    fc_cluster_list = []
    fc_creator_list = []
    fc_vector_list = []
    for i in range(len(fc_emb_df)):
        fc_sentence = fc_emb_df["Sentence"][i]
        fc_keyword_list = get_results_for_one_text(fc_sentence[:OUTPUT_TOKENS], PROMPT_TEXT_FC)
        for kw_i in range(len(fc_keyword_list)):
            kw = fc_keyword_list[kw_i].replace("・", "").replace(" ", "").replace("　", "")           
            # if (len(kw) >= 3) & (len(kw) < 50):
            if (len(kw) >= 3):
                fc_keyword_all.append(kw)
                fc_users.append("FC")
                fc_cluster_list.append(i)
                fc_vector_list.append(get_sentence_vector(kw))
                fc_creator_list.append("")
        flt_dl = dl_emb_df[dl_emb_df["Cluster"] == i].reset_index(drop=True)
        for dl_i in range(len(flt_dl)):
            dl_sentence = flt_dl["Sentence"][dl_i]
            dl_user = flt_dl["DataType"][dl_i]
            dl_creator = flt_dl["Creator"][dl_i]
            if dl_user == "DLData":
                kw = dl_sentence.strip().replace("・", "").replace(" ", "").replace("　", "")
                # if (len(kw) >= 3) & (len(kw) < 50):
                if (len(kw) >= 3):
                    fc_keyword_all.append(kw)
                    fc_users.append(dl_user)
                    fc_cluster_list.append(i)
                    fc_vector_list.append(get_sentence_vector(kw))
                    fc_creator_list.append(dl_creator)
            elif dl_user == "DLTokyoData":
                kw = dl_sentence.strip().replace("・", "").replace(" ", "").replace("　", "")
                # if (len(kw) >= 3) & (len(kw) < 50):
                if (len(kw) >= 3):
                    fc_keyword_all.append(kw)
                    fc_users.append(dl_user)
                    fc_cluster_list.append(i)
                    fc_vector_list.append(get_sentence_vector(kw))
                    fc_creator_list.append(dl_creator)
            elif dl_user == "UserAns":
                kw = dl_sentence.strip().replace("・", "").replace(" ", "").replace("　", "")
                # if (len(kw) >= 3) & (len(kw) < 50):
                if (len(kw) >= 3):
                    fc_keyword_all.append(kw)
                    fc_users.append(dl_user)
                    fc_cluster_list.append(i)
                    fc_vector_list.append(get_sentence_vector(kw))
                    fc_creator_list.append(dl_creator)
            else:
                dl_keyword_list = get_results_for_one_text(dl_sentence[:OUTPUT_TOKENS], PROMPT_TEXT_OTR)
                for kw_i in range(len(dl_keyword_list)):
                    kw = dl_keyword_list[kw_i].replace("・", "").replace(" ", "").replace("　", "")
                    # if (len(kw) >= 3) & (len(kw) < 50):
                    if (len(kw) >= 3):
                        fc_keyword_all.append(kw)
                        fc_users.append(dl_user)
                        fc_cluster_list.append(i)
                        fc_vector_list.append(get_sentence_vector(kw))
                        fc_creator_list.append(dl_creator)

    fc_df_base = pd.DataFrame({
        'Keyword': fc_keyword_all, 
        'User': fc_users, 
        'Cluster' : fc_cluster_list, 
        'Creator' : fc_creator_list,
        'Vector' : fc_vector_list
    })

    vectors = []
    for i in range(len(fc_df_base)):
        vectors.append(np.array(fc_df_base["Vector"][i]))
    norm_vectors = normalize(vectors)
    fc_df_base["NormVector"] = [nv for nv in norm_vectors]
    fc_df_base["UserCluster"] = [f"{usr}{clst}" for usr, clst in zip(fc_df_base["User"], fc_df_base["Cluster"])]
    
    # FC&DL両方
    fc_df_all = fc_df_base.copy()
    key_pair_list_all = list(itertools.combinations(fc_df_all["Keyword"], 2))
    key1_list = []
    key2_list = []
    sim_list = []
    for i in range(len(key_pair_list_all)):
        key1 = key_pair_list_all[i][0]
        key2 = key_pair_list_all[i][1]
        if fc_df_all[fc_df_all["Keyword"] == key1].reset_index(drop=True)["UserCluster"][0] != fc_df_all[fc_df_all["Keyword"] == key2].reset_index(drop=True)["UserCluster"][0]:
            vec1 = fc_df_all[fc_df_all["Keyword"] == key1].reset_index(drop=True)["NormVector"][0]
            vec2 = fc_df_all[fc_df_all["Keyword"] == key2].reset_index(drop=True)["NormVector"][0]
            sim_key1_key2 = get_similarity(vec1, vec2)
            key1_list.append(key1)
            key2_list.append(key2)
            sim_list.append(sim_key1_key2)

    key_pair_df_all = pd.DataFrame({
        'key1': key1_list, 
        'key2': key2_list, 
        'similarity': sim_list
    })
    # print(len(key_pair_df_all))

    # HTMLを作成
    # create_fcdl_network(fc_df_onlyfc, key_pair_df_onlyfc, False, seg, "onlyfc", fc_emb_df, network_num, fc_df)
    create_fcdl_network(fc_df_all, key_pair_df_all, False, seg, "fcdl", fc_emb_df, network_num, fc_df)
    # create_fcdl_network(fc_df_onlyfc, key_pair_df_onlyfc, True, seg, "onlyfc_img", fc_emb_df, network_num, fc_df)
    # create_fcdl_network(fc_df_all, key_pair_df_all, True, seg, "fcdl_img", fc_emb_df, network_num, fc_df)
def create_fcdl_network(fc_df, key_pair_df, image_flag, fname, tp, emb_df, network_num, orig_df):
    flag_master = pd.DataFrame({
        "User": ["FC", "DLData", "DLTokyoData", "UserAns"], 
        "Flag": [1, 2, 3, 4], 
        "NodeColor": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"], 
        "EdgeColor": ["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4"]
    })
    
    connect_df1 = fc_df[["Keyword", "UserCluster"]]
    connect_df1.columns = ["key1", "key2"]
    connect_df2 = key_pair_df.sort_values(["similarity"], ascending=False).head(network_num).reset_index(drop=True)[["key1", "key2"]]
    
    if tp == "fcdl":
        fcdl_k1 = []
        fcdl_k2 = []
        for dttp in ["DLData", "DLTokyoData","UserAns"]:
            for i in range(len(emb_df)):
                fcdl_k1.append(f"FC{str(i)}")
                fcdl_k2.append(f"{dttp}{str(i)}")
        connect_df3 = pd.DataFrame({
            'key1': fcdl_k1, 
            'key2': fcdl_k2
        })
        connect_df4 = orig_df[["Keyword", "User"]].copy()
        connect_df4 = connect_df4[connect_df4["Keyword"] != ""]
        connect_df4 = connect_df4[connect_df4["User"] != ""]
        connect_df4.columns = ["key1", "key2"]

        connect_df5 = fc_df[["Keyword", "Creator"]].copy()
        connect_df5 = connect_df5[connect_df5["Keyword"] != ""]
        connect_df5 = connect_df5[connect_df5["Creator"] != ""]
        connect_df5.columns = ["key1", "key2"]

        connect_df = pd.concat([connect_df1, connect_df2, connect_df3, connect_df4, connect_df5]).dropna().drop_duplicates().reset_index(drop=True)
    else:
        connect_df = pd.concat([connect_df1, connect_df2]).drop_duplicates().reset_index(drop=True)
    
    onlyfc_gp = pd.merge(fc_df[["Keyword", "User"]], flag_master[["User", "Flag"]], on="User", how="left")[["Keyword", "Flag"]]
    onlyfc_gp.columns = ["NodeName", "Flag"]
    
    node_df1 = connect_df[["key1"]]
    node_df1.columns = ["NodeName"]
    node_df2 = connect_df[["key2"]]
    node_df2.columns = ["NodeName"]
    node_df = pd.concat([node_df1, node_df2]).drop_duplicates().reset_index(drop=True)
    node_df["NodeNum"] = [1+i for i in range(len(node_df))]
    node_df = pd.merge(node_df, onlyfc_gp, on="NodeName", how="left")
    node_df["Flag"] = node_df["Flag"].fillna(0)

    if image_flag:
        img_list = []
        for nd_i in range(len(node_df)):
            search_word = node_df["NodeName"][nd_i]
            print(search_word)
            img = getImageUrl(API_KEY, CUSTOM_SEARCH_ENGINE, search_word)
            img_list.append(img)
        node_df['ImageURL'] = img_list
    
    connect_df = pd.merge(connect_df, node_df, left_on="key1", right_on="NodeName", how="left")[["key1", "key2", "NodeNum", "Flag"]]
    connect_df.columns = ["key1", "key2", "NodeNum1", "Flag1"]
    connect_df = pd.merge(connect_df, node_df, left_on="key2", right_on="NodeName", how="left")[["key1", "key2", "NodeNum1", "Flag1", "NodeNum", "Flag"]]
    connect_df.columns = ["key1", "key2", "NodeNum1", "Flag1", "NodeNum2", "Flag2"]
    # connect_df["Flag1"] = [1 if k1[:2] == "FC" else flg for flg, k1 in zip(connect_df["Flag1"], connect_df["key1"])]
    # connect_df["Flag2"] = [1 if k2[:2] == "FC" else flg for flg, k2 in zip(connect_df["Flag2"], connect_df["key2"])]
    connect_df = connect_df.fillna(0)
    connect_df["FCConnect"] = connect_df["Flag1"] * connect_df["Flag2"]

    # ネットワークのインスタンス生成
    network = Network(
        height="1000px",  # デフォルト "500px"
        width="2000px",  # デフォルト "500px"
        notebook=True,  # これをTrueにしておくとjupyter上で結果が見れる
        bgcolor='#ffffff',  # 背景色。デフォルト "#ffffff"
        directed=False,  # Trueにすると有向グラフ。デフォルトはFalseで無向グラフ
    )

    # add_node でノードを追加
    for i in range(len(node_df)):
        nd1_id = int(node_df['NodeNum'][i])
        nd1_name = node_df['NodeName'][i]
        nd1_flag = node_df['Flag'][i]

        if nd1_flag == 1:
            nd1_color = flag_master["EdgeColor"][0]
        elif nd1_flag == 2:
            nd1_color = flag_master["EdgeColor"][1]
        elif nd1_flag == 3:
            nd1_color = flag_master["EdgeColor"][2]
        elif nd1_flag == 4:
            nd1_color = flag_master["EdgeColor"][3]
        elif nd1_name[:2] == "FC":
            nd1_color = flag_master["NodeColor"][0]
        elif nd1_name[:6] == "DLData":
            nd1_color = flag_master["NodeColor"][1]
        elif nd1_name[:6] == "DLTokyoData":
            nd1_color = flag_master["NodeColor"][2]
        elif nd1_name[:7] == "UserAns":
            nd1_color = flag_master["NodeColor"][3]
        else:
            nd1_color = "#e5c494"
        
        if image_flag:
            nd1_image = node_df['ImageURL'][i]
            network.add_node(n_id=nd1_id, label=nd1_name, shape='image', image =nd1_image)
        else:
            network.add_node(n_id=nd1_id, label=nd1_name, color=nd1_color)

    for i in range(len(connect_df)):
        nd1_id = int(connect_df['NodeNum1'][i])
        nd2_id = int(connect_df['NodeNum2'][i])
        nd_flag = connect_df['FCConnect'][i]

        if nd_flag == 1:
            edge_color = "#2c7bb6"
        else:
            edge_color = "#1a9641"

        # network.add_edge(nd1_id, nd2_id, color=edge_color, width = edge_width)
        network.add_edge(nd1_id, nd2_id, color=edge_color, width=0.1)
    # 指定したファイル名でHTMLを出力。
    network.show(f"D:\\code\\FCDL_にぎわい\\htmls\\fc_{fname}_{tp}.html")
    return 0


clustering_and_create_network(fc_all, 
                              embeddings_dls, 
                              embeddings_dl_tokyo,
                              dl_all,
                              dl_tokyo, 
                              "nigiwai_V3", 
                              4, 
                              5)

