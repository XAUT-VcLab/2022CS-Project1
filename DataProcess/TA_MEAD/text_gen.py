import pandas as pd
import random


def generate_text_for_video(
        video_token,
        AU_value_df,
        emotion_table,
        AU_description_df,
        intensity_df,
        use_intensity=True,
        AU_intensity_split_df=None,
        use_emotion=True,
        use_AU=True,
):
    if not (use_emotion or use_AU):
        raise NotImplementedError("Have not test use_emo=False and use_AU=False")

    video_df = gen_video_df(video_token, AU_value_df, AU_intensity_split_df)
    text_list = video_df2text(
        video_df,
        1,
        emotion_table,
        intensity_df,
        AU_description_df,
        use_intensity=use_intensity,
        use_emotion=use_emotion,
        use_AU=use_AU,
    )

    return text_list[0]


def gen_video_df(video_token, AU_value_df, AU_intensity_split_df):
    person_id, direction, emotion, level, video_idx = video_token.split("_")
    ### Gender
    if person_id[0] == "M":
        gender_df_value = 0.0
    elif person_id[0] == "W":
        gender_df_value = 1.0
    else:
        raise ValueError

    ### AU

    video_au_entry = AU_value_df[AU_value_df["video"] == video_token]
    activated_au_filter = video_au_entry.iloc[:, 1:] > 1
    activated_au_list = list(video_au_entry.columns[1:][activated_au_filter.iloc[0]])

    remove_AU_list = ["AU25", "AU26"]
    for remove_AU in remove_AU_list:
        if remove_AU in activated_au_list:
            activated_au_list.remove(remove_AU)

    ### Intensity
    activated_au_entry = video_au_entry[activated_au_list]
    intensity = get_intensity_df(activated_au_entry, AU_intensity_split_df)

    ### generate dataframe
    df_entry_list = []
    for au_item in activated_au_list:
        action_unit_df_value = float(au_item[2:])

        new_df_entry = {
            "Gender": gender_df_value,
            "Emotion": emotion,
            "ActionUnit": action_unit_df_value,
            "Intensity": intensity[au_item].iloc[0],
            "EmotionLevel": level,
        }
        df_entry_list.append(new_df_entry)
        # video_df.append(new_df_entry, ignore_index=True)
    if len(df_entry_list) > 0:
        video_df = pd.DataFrame(df_entry_list)
    elif len(df_entry_list) == 0:
        video_df = pd.DataFrame(
            {
                "Gender": [gender_df_value],
                "Emotion": [emotion],
                "ActionUnit": [None],
                "Intensity": [None],
                "EmotionLevel": [level],
            }
        )

    return video_df


def get_emotion_and_pattern(key, level, emotion_table):
    emotion_dict = emotion_table[key][level]
    emo_and_patt_list = list()
    for patt, emotion_list in emotion_dict.items():
        for emo in emotion_list:
            emo_patt_pair = (emo, patt)
            emo_and_patt_list.append(emo_patt_pair)

    emo, patt = random.choice(emo_and_patt_list)

    if patt not in emotion_table["valid_patterns"]:
        raise ValueError(f"cur pattern: {patt}, valid patterns: {emotion_table['valid_patterns']}")

    return emo, patt


def textParam(dataframe, emotion_table, dfIntensity, dfActionUnit):
    # dfEmotion = pd.read_csv('Emotion.csv')
    # dfIntensity = pd.read_csv('Intensity.csv')
    # dfActionUnit = pd.read_csv('ActionUnit.csv')

    # gender
    if int((set(dataframe["Gender"])).pop()) == 0:
        Gender = "man"

    elif int((set(dataframe["Gender"])).pop()) == 1:
        Gender = "woman"

    # emotion
    emotion_key = set(dataframe["Emotion"]).pop()
    emotion_level = set(dataframe["EmotionLevel"]).pop()
    Emotion, Emotion_pattern = get_emotion_and_pattern(emotion_key, emotion_level, emotion_table)

    # TODO:
    if dataframe["ActionUnit"].iloc[0] is None:
        return Gender, Emotion, Emotion_pattern, None, None, None, None

    # action unit with intensity
    Intensity = []
    AUnoun = []
    AUadj = []
    AUI = dataframe[["ActionUnit", "Intensity"]]
    numau = []  ### for constraint
    for i in range(len(AUI)):
        aui_comb = dict(AUI.iloc[i])
        AUAU = aui_comb["ActionUnit"]

        if not str(AUAU)[-1] == "0":
            raise ValueError
        else:
            numActionUnit = int(aui_comb["ActionUnit"])
        numIntensity = int(round(aui_comb["Intensity"]))
        if numIntensity < 0:
            numIntensity = 0

        numau.append(numActionUnit)  ### for constraint

        # action unit with noun and adjective(2 types)
        AUnoun.append(dfActionUnit.iloc[2][str(numActionUnit)])
        listAUadj = list(dfActionUnit.iloc[-2:][str(numActionUnit)])
        if type(listAUadj[1]) == float:
            AUadj.append(listAUadj[0])
        else:
            randAUadj = random.choice(listAUadj)
            AUadj.append(randAUadj)

        # intensity
        if numIntensity == 0:
            randIntensity = None
            Intensity.append(randIntensity)
        else:
            listIntensity = list(dfIntensity[str(numIntensity)])
            listIntensity = [Int for Int in listIntensity if str(Int) != "nan"]  # remove nan
            randIntensity = random.choice(listIntensity)
            Intensity.append(randIntensity)

    return Gender, Emotion, Emotion_pattern, Intensity, AUnoun, AUadj, numau


def get_text_emotion_part(gender, emotion, pattern, emotion_table):
    text = None
    emotion = emotion.lower()
    if pattern == "feel":
        text = f"{gender} feels {emotion}"
    elif pattern == "show":
        show_verb = random.choice(emotion_table["pattern"]["show"]["show_words"])
        text = f"{gender} {show_verb} {emotion}"
    elif pattern == "is_in":
        text = f"{gender} is in {emotion}"
    elif pattern == "attr":
        text = f"{emotion} {gender}"
    else:
        raise ValueError(f"Invalid pattern {pattern}")

    if text[0] in {"a", "e", "i", "o", "u"}:
        text = f"An {text}"
    else:
        text = f"A {text}"

    return text


def video_df2text(
        dataframe,
        numtext,
        emotion_table,
        intensity_df,
        AU_description_df,
        use_intensity=True,
        use_emotion=True,
        use_AU=True,
        maintain_AU_order=False,
        maintain_adj_noun_order=False,
):
    """
    :param param: [Gender, Emotion, Intensity, AUnoun, AUadj]
    :return: number of texts
    """

    if not use_intensity:
        dataframe["Intensity"] = 0.0

    tlist = []
    for t in range(numtext):

        Gender, Emotion, emotion_pattern, Intensity, AUnoun, AUadj, numau = textParam(
            dataframe, emotion_table, intensity_df, AU_description_df
        )

        Emotion = Emotion.lower()
        if use_emotion or AUnoun is None:
            txt = get_text_emotion_part(Gender, Emotion, emotion_pattern, emotion_table)
        else:
            txt = f"A {Gender}"

        if AUnoun is None or not use_AU:
            txt = txt + "."
            tlist.append(txt)
            continue

        ### add preposition
        preposition = ["speaks with"]
        prep = random.choice(preposition)
        if emotion_pattern in {"feel", "is_in", "show"}:
            txt = f"{txt} and {prep} "
        elif emotion_pattern in {"attr"}:
            txt = f"{txt} {prep} "
        else:
            raise ValueError(f"Unknown pattern for preposition generation: {emotion_pattern}")

        ### add AU part
        if not maintain_AU_order:
            # change order of AU
            shuffleAU = list(zip(Intensity, AUnoun, AUadj))
            random.shuffle(shuffleAU)
            Intensity, AUnoun, AUadj = zip(*shuffleAU)

        if not maintain_adj_noun_order:
            for i in range(len(AUnoun)):
                if random.random() > 0.5:
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + " " + AUadj[i] + "."
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = AUnoun[i] + " " + AUadj[i] + " "
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + AUnoun[i] + " " + AUadj[i] + "."
                        else:
                            txt2 = AUnoun[i] + " " + AUadj[i] + ", "

                    else:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + "."
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + " "
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + "."
                        else:
                            txt2 = AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + ", "
                    txt = txt + txt2

                else:
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = AUadj[i] + " " + AUnoun[i] + "."
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = AUadj[i] + " " + AUnoun[i] + " "
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + AUadj[i] + " " + AUnoun[i] + "."
                        else:
                            txt2 = AUadj[i] + " " + AUnoun[i] + ", "

                    else:
                        if len(AUnoun) == 1:
                            txt2 = Intensity[i] + " " + AUadj[i] + " " + AUnoun[i] + "."
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = Intensity[i] + " " + AUadj[i] + " " + AUnoun[i] + " "
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + Intensity[i] + " " + AUadj[i] + " " + AUnoun[i] + "."
                        else:
                            txt2 = Intensity[i] + " " + AUadj[i] + " " + AUnoun[i] + ", "

                    txt = txt + txt2

        else:
            if random.random() > 0.5:
                for i in range(len(AUnoun)):
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + " " + AUadj[i] + "."
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + AUnoun[i] + " " + AUadj[i] + "."
                        else:
                            txt2 = AUnoun[i] + " " + AUadj[i] + ", "

                    else:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + "."
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + "."
                        else:
                            txt2 = AUnoun[i] + " " + Intensity[i] + " " + AUadj[i] + ", "
                    txt = txt + txt2

            else:
                for i in range(len(AUnoun)):
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + AUadj[i] + " " + AUnoun[i] + "."
                        else:
                            txt2 = AUadj[i] + " " + AUnoun[i] + ", "

                    else:
                        if i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = "and " + Intensity[i] + " " + AUadj[i] + " " + AUnoun[i] + "."
                        else:
                            txt2 = Intensity[i] + " " + AUadj[i] + " " + AUnoun[i] + ", "

                    txt = txt + txt2

        tlist.append(txt)

    return tlist


def get_intensity_df(activated_au_entry, AU_intensity_split_df):
    """
    input:
            AU1	        AU2	        AU5
    30355	4.056374	2.860973	2.436523

    output:
            AU1	        AU2	        AU5
    30355	3.0	        2.0	        3.0

    Args:
        activated_au_entry (_type_): _description_
        AU_intensity_split_df (_type_): _description_
    """
    intensity_df = activated_au_entry.copy()
    for AU_item in activated_au_entry.columns:
        thr_1, thr_2 = AU_intensity_split_df[AU_item]
        cur_au_value = activated_au_entry[AU_item].iloc[0]
        if cur_au_value < thr_1:
            intensity_df[AU_item] = 1.0
        elif thr_1 <= cur_au_value < thr_2:
            intensity_df[AU_item] = 2.0
        elif cur_au_value >= thr_2:
            intensity_df[AU_item] = 3.0
        else:
            raise ValueError(f"{cur_au_value}, {thr_1}, {thr_2}")

    return intensity_df


if __name__ == "__main__":
    import json

    # video_token = "W014_front_surprised_level1_001"
    # video_token = "M007_front_angry_level3_001"
    # video_token = "M028_front_happy_level3_001"
    video_token = "W028_front_angry_level3_001"
    # video_token = "M009_front_contempt_level2_006"
    # video_token = "M003_front_disgusted_level1_010"
    AU_value_df = pd.read_csv("AU_value.csv")
    import yaml

    with open("complex_emotion.yaml", "r") as f:
        emotion_table = yaml.full_load(f)
    AU_description_df = pd.read_csv("ActionUnit.csv")
    intensity_df = pd.read_csv("Intensity.csv")
    AU_intensity_split_df = pd.read_csv("AU_intensity_split.csv")
    with open("emotion2idx.json", "r") as f:
        emotion2idx = json.load(f)

    text = generate_text_for_video(
        video_token,
        AU_value_df,
        emotion_table,
        AU_description_df,
        intensity_df,
        use_intensity=True,
        AU_intensity_split_df=AU_intensity_split_df,
        use_emotion=True,
        use_AU=True,
    )
    print(text)
