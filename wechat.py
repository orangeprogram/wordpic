from wxpy import *
import re
import jieba
import pandas as pd
import numpy
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

# 初始化机器人，扫码登陆
bot = Bot()

# 获取所有好友
my_friends = bot.friends()

# 使用一个字典统计好友男性和女性的数量
sex_dict = {'male': 0, 'female': 0}

# 统计性别
for friend in my_friends:

    if friend.sex == 1:
        sex_dict['male'] += 1
    elif friend.sex == 2:
        sex_dict['female'] += 1

# 使用一个字典统计各省好友数量
province_dict = {'北京': 0, '上海': 0, '天津': 0, '重庆': 0,
                 '河北': 0, '山西': 0, '吉林': 0, '辽宁': 0, '黑龙江': 0,
                 '陕西': 0, '甘肃': 0, '青海': 0, '山东': 0, '福建': 0,
                 '浙江': 0, '台湾': 0, '河南': 0, '湖北': 0, '湖南': 0,
                 '江西': 0, '江苏': 0, '安徽': 0, '广东': 0, '海南': 0,
                 '四川': 0, '贵州': 0, '云南': 0,
                 '内蒙古': 0, '新疆': 0, '宁夏': 0, '广西': 0, '西藏': 0,
                 '香港': 0, '澳门': 0}

# 统计省份
for friend in my_friends:
    if friend.province in province_dict.keys():
        province_dict[friend.province] += 1

# 为了方便数据的呈现，生成JSON Array格式数据
data = []
for key, value in province_dict.items():
    data.append({'name': key, 'value': value})


def write_txt_file(path, txt):

    with open(path, 'a', encoding='gb18030', newline='') as f:
        f.write(txt)


# 统计签名
for friend in my_friends:
    print(friend.signature)
    pattern = re.compile(r'[一-龥]+')
    filterdata = re.findall(pattern, friend.signature)
    write_txt_file('signatures.txt', ''.join(filterdata))


def read_txt_file(path):

    with open(path, 'r', encoding='gb18030', newline='') as f:
        return f.read()


content = read_txt_file('signatures.txt')
segment = jieba.lcut(content)
words_df = pd.DataFrame({'segment': segment})

stopwords = pd.read_csv("stopwords.txt", index_col=False,
                        quoting=3, sep=" ", names=['stopword'], encoding='utf-8')
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]

words_stat = words_df.groupby(by=['segment'])[
    'segment'].agg({"计数": numpy.size})
words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)
print(words_df)
print(words_stat)

# 设置词云属性
color_mask = imread('background.jfif')
wordcloud = WordCloud(font_path="simhei.ttf", # 设置字体可以显示中文
                      background_color="white",# 背景颜色
                      max_words=100, # 词云显示的最大词数
                      mask=color_mask, # 设置背景图片
                      max_font_size=100,# 字体最大值
                      random_state=42,
                      # 设置图片默认的大小,但是如果使用背景图片的话,                                                   # 那么保存的图片大小将会按照其大小保存,margin为词语边缘距离
                      width=1000, height=860, margin=2,
                      )

# 生成词云, 可以用generate输入全部文本,也可以我们计算好词频后使用generate_from_frequencies函数
word_frequence = {x[0]: x[1]for x in words_stat.head(100).values}
print(word_frequence)
word_frequence_dict = {}
for key in word_frequence:
    word_frequence_dict[key] = word_frequence[key]

wordcloud.generate_from_frequencies(word_frequence_dict)
# 从背景图片生成颜色值  
image_colors = ImageColorGenerator(color_mask)
# 重新上色
wordcloud.recolor(color_func=image_colors)
# 保存图片
wordcloud.to_file('output.png')
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
