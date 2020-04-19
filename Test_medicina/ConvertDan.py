import copy
# Class - для подготовки данных с предварительной обработкой
class ConverDan(object):
    import numpy as np
    import os
    def __init__(self, *args, **kwargs):
        print(" -= Class-> ConverDan Dan =-")

# Функция обрабатывает первоначальные данные из фалов
# И формирует на выходе данные _complaint, _lsWork, lsPhrase
# _complaint - словарь с обращениями как в файлах
# _lsWork - слова в каждом обращение
# lsPhrase - фразы в каждом обращение -> делал c "прицелом" на Embedding
    def Basic_processing(self, sbasa):
        import re
        import pymorphy2

        morph = pymorphy2.MorphAnalyzer()
        reg_simvol = re.compile('[^a-zA-Z0-9а-яА-Я]') 

        _complaint = dict()      # жалобы
        _n_complaint = 0         # номер жалобы
        _sum_work    = 0         # сумма слов 1 обращении 
        _lsWork = []             # слова
        _lsPhrase = []          # фраза

        sbasa = [x.strip().lower() for x in sbasa]

        for it in sbasa:
            _s = reg_simvol.sub(' ', it).strip()
            print(_s)
            _n_line = len(_s)

            if _n_line == 0: continue

            if (_n_line < 4) & _s.isdigit():   #	Состоит ли строка из цифр  S.isalnum()	Состоит ли строка из цифр или букв
                _n_complaint = int(_s.strip())
                _sum_work    = 0
                _complaint[_n_complaint] = []
                _dan_one_complaint      =  ConverDan.__creat_dict_one_complaint(self)
            else:
                __s02 = [morph.parse(x)[0].normal_form for x in [x.strip() for x in _s.split(" ")] if len(x)>1]
                __ns02 = len(__s02)
                _lsWork.extend(__s02)
                __s03 = ' '.join ([x for x in __s02])
                _lsPhrase += [(__s03, __ns02)]
                _sum_work += __ns02
                _dan_one_complaint["Phrase"] += [(__s03, __ns02)]
                _dan_one_complaint["Work"].extend(__s02)
                _dan_one_complaint["CountWork"] += __ns02
                _complaint[_n_complaint]=copy.deepcopy(_dan_one_complaint)

#!!! Это важно оставляем только уникальные фразы, для дальнейшей генерпции текста
        res = list(dict.fromkeys(_lsPhrase))    # 
        lsPhrase = {i:res[i] for i in range(len(res)) if len(res[i])>0}
        return   _complaint, _lsWork, lsPhrase

# формируем одно обращение в котором записываем кол-во слов, используемые слова и фразы
    def __creat_dict_one_complaint(self):
        _dan_one_complaint  = dict()
        _dan_one_complaint["Phrase"] = []
        _dan_one_complaint["Work"] =[]
        _dan_one_complaint["CountWork"] =0
        return _dan_one_complaint

# проверка как получилась база
    def StatDanComplaint(self, all_basa:dict):
        _stat_dan_all = dict()

        def __print_basa():
            print("\n Разбираем значения в базе")
            for key, val in _stat_dan_all.items():
                print("\n == КЛАСС -> {x0}    {x1} ".format(x0= key, x1=val["name"]))
                print(" - кол-во обращений = {x0}  ".format(x0= val["count_complaint"]))
                print(" - кол-во фраз = {x0}  ".format(x0= val["count_phrase"]))
                print(" - кол-во слов = {x0}  ".format(x0= val["count_work"]))
                print(" - седнее кол-во слов в одном обращении = {x0}  ".format(x0= val["count_work"]//val["count_complaint"]))
                

        for key, val in all_basa.items():
            print(" обработка данных -> {x0}".format(x0= val[0]))
            d = val[1]
            _stat_dan_one = dict()
            _stat_dan_one["name"]=val[0]
            _stat_dan_one["count_complaint"] = len(val[1]["complaint"])
            _stat_dan_one["count_phrase"] = len(val[1]["lsPhrase"])
            _stat_dan_one["count_work"] = len(val[1]["lsWork"])
            _stat_dan_all[len(_stat_dan_all)+1] = _stat_dan_one

        __print_basa()
        return _stat_dan_all

# считаем нужное кол-во слов по каждой группе
    def calc_complaint_and_work(self, _stat_dan_all, _ncomplaint=500, _word =25):
            print("\n Расчитаем сколько нужно слов")
            print("    при {x0} обращений - train  и {x1} - test по каждому заболеванию".format(x0=_ncomplaint, x1=int(_ncomplaint*0.2)))
            print("    положем что обращение 25 слов")
            all_word_train = _ncomplaint*_word
            all_word_test = int(_ncomplaint*0.2)*_word

            for key, val in _stat_dan_all.items():
                print("\n == КЛАСС -> {x0}    {x1} ".format(x0= key, x1=val["name"]))

                _count_complaint = val["count_complaint"]
                _ost = _ncomplaint-_count_complaint
                _ost_word = all_word_train  - val["count_work"]
                _sr_word = int(_ost_word/_ost)
                
                print(" Дополнительно сгенерить: ")
                print(" - кол-во обращений = {x0}  ".format(x0= _ost))
                print(" - примерное кол-во слов в одном обращении = {x0}  ".format(x0= _sr_word))

# создаем базы обращений train_test
# train из расчета 500 обращений по 30 слов - подставляю фразы и кол-во слов задаю генератором
# test формирую из расчета 25% от train 
    def complaint_train_test(self, danx, _ncomplaint=500, _word_basa=25):
        import random
        dan = danx[1]
        dan["name"] = danx[0];
        _complaint = dan["complaint"] 
        _lsWork =  dan["lsWork"] 
        _lsPhrase = dan["lsPhrase"]
        _ilsPhrase=len(_lsPhrase)-1
        _train = copy.deepcopy(_complaint)
        _test = dict()

        def one_complaint(min_word):
            _dan_one_complaint  = ConverDan.__creat_dict_one_complaint(self)
            while _dan_one_complaint["CountWork"] < min_word:
                __dan= _lsPhrase[random.randint(0, _ilsPhrase)]
                _dan_one_complaint["CountWork"] += __dan[1]
                _dan_one_complaint["Phrase"] += [__dan]
                _dan_one_complaint["Work"].extend(__dan[0].split(" "))
            return _dan_one_complaint

        def __calc_word(_all_word_train, _n_lsWork, _calc_count, step):
            __word = int((_all_word_train- _n_lsWork)//_calc_count)
            __word_min = __word - step
            __word_max = __word + step
            return __word_min, __word_max

        def __calc_train_test(_dan:dict, _ncomplaint):
            _all_word_train = _ncomplaint*_word_basa
            if len(_dan) == 0:
                _n_lsWork = 0
                _word = 25
            else:
                _n_lsWork = len(_lsWork)
                _word = int((_all_word_train- _n_lsWork)//(_ncomplaint - len(_dan)))

            _word_min = _word -5
            _word_max = _word +5

            while _n_lsWork < _all_word_train:
                _n_rand_word = random.randint(_word_min, _word_max)
                _dan_one_complaint = one_complaint(_n_rand_word)

                _n_complaint =len(_dan)+1
                _dan[_n_complaint] = copy.deepcopy(_dan_one_complaint)
                _lsWork.extend(_dan_one_complaint["Work"])
                _n_lsWork = len(_lsWork)
                _calc_count =_ncomplaint-_n_complaint

                if _calc_count <=0:
                    break

                if _calc_count <=5:
                    _word_min, _word_max = __calc_word(_all_word_train, _n_lsWork, _calc_count, 1)
                    continue
                elif _calc_count <=10:
                    _word_min, _word_max = __calc_word(_all_word_train, _n_lsWork, _calc_count, 3)
                    continue
                elif _calc_count <=50:
                    _word_min, _word_max = __calc_word(_all_word_train, _n_lsWork, _calc_count, 4)
                    continue
            return _dan

        print("==>>==> Формируем данные для {x0} \n    -----> train ".format(x0=danx[0]))
        dan["train"] = copy.deepcopy(__calc_train_test(_train, _ncomplaint))

        print("  -----> test ")
        _lsWork =[]
        dan["test"] = copy.deepcopy(__calc_train_test(_test, int(_ncomplaint*0.25)))
        return dan


    def print_dan0(self):
        for key, val in self.category_d.items():
            print(" группа - {x0} ".format(x0=val["name"]))
            print("   отношение слов test/trean - {x0} % ".format(x0=round((val["test"]/(val["test"]+val["train"]))*100,2) ))
    
# преобразую train_test в длинную строку
    def Create_Train_Test(self, all_dan:dict):
        self.category = dict()
        self.category_d=dict()
        self.train_text=[]
        self.test_text=[]
        self.count_word_dict=dict()
        for key, val in all_dan.items():
            __count_word = 0
            __s0 = ""
            for key1, val1 in val["train"].items():
                __count_word += val1["CountWork"]
                __s0 =__s0 +" "+ str( ' '.join ([str(x) for x in val1["Work"]]))
            __s0 = ' '.join(__s0.split())

            self.category[key-1] = [val["name"]]
            self.category_d[key-1] = {"name":val["name"], "train":__count_word} 
            self.train_text +=[__s0]

        for key, val in all_dan.items():
            __count_word = 0
            __s0 = ""
            for key1, val1 in val["test"].items():
                __count_word += val1["CountWork"]
                __s0 =__s0 +" "+ str( ' '.join ([str(x) for x in val1["Work"]]))
            __s0 = ' '.join(__s0.split())

            d=self.category_d[key-1]
            d["test"]=__count_word
            self.category_d[key-1] = d
            self.test_text +=[__s0]
        ConverDan.print_dan0(self)
        ConverDan.SaveDanALLtext(self)

    def SaveDanALLtext(self):
        self.DanALL_text = dict()
        self.DanALL_text["category"] = self.category_d
        self.DanALL_text["train_text"] = self.train_text
        self.DanALL_text["test_text"] = self.test_text

    def LoadDanALLtext(self, dan, print_dis=False):
        self.category_d = dan["category"]
        self.train_text = dan["train_text"]
        self.test_text = dan["test_text"]
        self.category = {key: val["name"] for key, val in self.category_d.items()}
        if print_dis :  self.print_dan0()

# формирую данные для передачу в пейронную сеть
    def Form_dan_to_model(self, _rwfiles, _dan):
    #    maxWordsCount, xLen, step = 50, 50, 10
        self.maxWordsCount, self.xLen, self.step = _dan[0], _dan[1], _dan[2]

        z =  _rwfiles.LoadFilePckle("DanALL_text")
        self.LoadDanALLtext(z)

        trein_text, test_text, tokenizer =  self.tokenizer_train_test(self.maxWordsCount)   # maxWordsCount=100
        #Формируем обучающую и тестовую выборку
        xTrain, yTrain = self.createSetsMultiClasses(self.trainWordIndexes, self.xLen, self.step) #извлекаем обучающую выборку
        xTest, yTest = self.createSetsMultiClasses(self.testWordIndexes, self.xLen, self.step)    #извлекаем тестовую выборку

        #Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
        zzz=xTrain.tolist()
        xTrain01 = tokenizer.sequences_to_matrix(xTrain.tolist())                       # подаем xTrain в виде списка чтобы метод успешно сработал
        xTest01 = tokenizer.sequences_to_matrix(xTest.tolist())                         # подаем xTest в виде списка чтобы метод успешно сработал
        self.tokenizer = tokenizer
        return xTrain01, yTrain, xTest01, yTest, self.category 

    def tokenizer_train_test(self, maxWordsCount=100):
        from keras.preprocessing.text import Tokenizer
        self.maxWordsCount = maxWordsCount
        #для этого воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
        tokenizer = Tokenizer(num_words=self.maxWordsCount, filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0' \
                              , lower=True, split=' ', char_level=False)

        tokenizer.fit_on_texts(self.train_text)                                 # "скармливаем" наши тексты, т.е даём в обработку методу, который соберет словарь частотности
        self.trainWordIndexes = tokenizer.texts_to_sequences(self.train_text)   # обучающие тесты в индексы
        self.testWordIndexes  = tokenizer.texts_to_sequences(self.test_text)    # проверочные тесты в индексы
        return self.train_text, self.test_text, tokenizer

    ###########################
    # Формирование обучающей выборки по листу индексов слов
    # (разделение на короткие векторы)
    ##########################
    def __getSetFromIndexes(self, wordIndexes, xLen, step):

      xSample, index = [], 0
      wordsLen = len(wordIndexes)
  
      #Идём по всей длине вектора индексов
      #"Откусываем" векторы длины xLen и смещаеммся вперёд на step
  
      while (index + xLen <= wordsLen):
        xSample.append(wordIndexes[index:index+xLen])
        index += step
      return xSample    

    def createSetsMultiClasses(self, wordIndexes, xLen, step): #функция принимает последовательность индексов, размер окна, шаг окна
      from tensorflow.keras import utils
      import numpy as np

      #Для каждого из N классов
      #Создаём обучающую/проверочную выборку из индексов
      nClasses = len(wordIndexes)       #задаем количество классов выборки
      classesXSamples = []              #здесь будет список размером "кол-во классов*кол-во окон в тексте*длину окна(например N по x0*xlen)"
      for wI in wordIndexes:            #для каждого текста выборки из последовательности индексов
        classesXSamples.append(ConverDan.__getSetFromIndexes(self, wI, xLen, step)) #добавляем в список очередной текст индексов, разбитый на "кол-во окон*длину окна" 

      #Формируем один общий xSamples
      xSamples = []                     #здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна(например 15779*1000)"
      ySamples = []                     #здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной 6"
  
      for t in range(nClasses):         #в диапазоне кол-ва классов(6)
        xT = classesXSamples[t]         #берем очередной текст вида "кол-во окон в тексте*длину окна"(например 1341*1000)
        for i in range(len(xT)):        #и каждое его окно
          xSamples.append(xT[i])        #добавляем в общий список выборки
    
        #Формируем ySamples по номеру класса
        currY = utils.to_categorical(t, nClasses)       #текущий класс переводится в вектор длиной 6 вида [0.0.0.1.0.0.]
        for i in range(len(xT)):        #на каждое окно выборки 
          ySamples.append(currY)        #добавляем соответствующий вектор класса

      xSamples = np.array(xSamples) #переводим в массив numpy для подачи в нейронку
      ySamples = np.array(ySamples) #переводим в массив numpy для подачи в нейронку
  
      return (xSamples, ySamples) #функция возвращает выборку и соответствующие векторы классов

    ###########################
    # Расспознаём тестовую выборку и выводим результаты
    ##########################
    #  ~~~~ Преобразуем строку  ~~~
    def convert_txt(self, _complaint):
        import re
        import pymorphy2

        morph = pymorphy2.MorphAnalyzer()
        reg_simvol = re.compile('[^a-zA-Z0-9а-яА-Я]') 

        __s02 = [morph.parse(reg_simvol.sub(' ', x).strip())[0].normal_form for x in (_complaint.lower()).split(" ")]

        s = ' '.join ([x for x in __s02 if len(x)>1])
        return s

    def recognizeMultiClass(self, model, xTest__):
        import numpy as np
        _text_complaint =  self.convert_txt(xTest__)

        self.tokenizer.fit_on_texts(_text_complaint)                              # "скармливаем" наши тексты, т.е даём в обработку методу, который соберет словарь частотности
        __text_WordIndexes = self.tokenizer.texts_to_sequences(_text_complaint)   #  в индексы
        _text_WordIndexes=[]
        z = [_text_WordIndexes.extend(x) for x in __text_WordIndexes]
        x_ = self.__getSetFromIndexes(_text_WordIndexes, self.xLen, self.step)
        __xTrain01 = self.tokenizer.sequences_to_matrix(x_)                       # подаем xTrain в виде списка чтобы метод успешно сработал

        currPred = model.predict(__xTrain01)
        prediction = np.argmax(currPred.sum(axis=0))

        return prediction, self.category[prediction]
