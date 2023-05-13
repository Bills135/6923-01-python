## 機器學習過程的個別有趣的程式  

- tensorflow  gpu 的測試程式  
- pytorch +cuda 測試  
- 課程的 demo code  
- Python 範例  
- 正在寫 Python 爬蟲抓 google 一直失敗 ...== 只抓到一頁  

miniconda3
jupyter notebook
python
tensorflow
pytorch
###
## RNN
循環神經網路（Recurrent neural network：RNN）是神經網路的一種。單純的RNN因為無法處理隨著遞歸，權重指數級爆炸或梯度消失問題，難以捕捉長期時間關聯；而結合不同的LSTM可以很好解決這個問題。  
https://zh.wikipedia.org/wiki/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C

## CNN 
卷積神經網絡CNN  
1）解決長遠距離的信息提取—把網絡做的更深一些，越深的捲積核將會有更大的感受野，從而捕獲更遠距離的特徵； 
2）為了防止文本中的位置信息丟失，NLP領域裡的CNN的發展趨勢是拋棄Pooling層，靠全卷積層來疊加網絡深度，並且在輸入部分加入位置編碼，人工將單詞的位置特徵加入到對應的詞向量中;   
3）網絡做深—殘差網絡—解決梯度消失問題—本質是加速信息流動，使簡單的信息傳輸可以有更簡單的路徑—網絡做深的同時確保有良好的性能；

## Transformer
![ai01](https://img-blog.csdnimg.cn/20210717145300296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb21laTk5OQ==,size_16,color_FFFFFF,t_70)
Transformer Block作為特徵提取器代替了之前提到的LSTM和CNN結構；  
3）通常使用的特徵提取結構(包括了Bert)主要是Encoder中的Transformer； 
理解Transformer在Encoder中是怎麼工作的； 
https://img-blog.csdnimg.cn/20210717150008341.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb21laTk5OQ==,size_16,color_FFFFFF,t_70

https://img-blog.csdnimg.cn/20210717150553143.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb21laTk5OQ==,size_16,color_FFFFFF,t_70

https://img-blog.csdnimg.cn/2021071715091428.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb21laTk5OQ==,size_16,color_FFFFFF,t_70

引用
https://blog.csdn.net/haomei999/article/details/118798615
# 6923-01-python  
