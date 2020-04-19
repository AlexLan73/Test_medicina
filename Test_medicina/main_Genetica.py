from ReadWriteFiles import ReadWriteFiles as RWFiles
from ConvertDan import ConverDan as CD
import copy
from Models import NWModel as NWM
import CFitGen
import GenDanDense
import WriteRead

# класс для чтения/записи данных на диск. Писал под Windows
# Данные должны храниться в такой структуре:
# ....\Болезни\
#               алкогольный гепатит\    <----- директория с файлами по данной болезни
#                       file1.txt
#                       file2.txt
# Система сама прочитает сколько групп и к каждой группе добавит файла с симптомами


def _form_basa_dan_treatment(_rwfiles, _cd):     # формируем базовые обращения
    all_dan = dict()
    ind = 1
    for item in _rwfiles._name_dir:
        print(item)     # z = _rwfiles.LoadFilePckle(item + "_basa")
        dan = dict()    #        complaint, lsWork, lsPhrase = _cd.Basic_processing(_rwfiles.ReadTextBasa(item))
        dan["complaint"], dan["lsWork"], dan["lsPhrase"] = _cd.Basic_processing(_rwfiles.ReadTextBasa(item))    #        complaint, lsWork, lsPhrase = _cd.Basic_processing(_rwfiles.ReadTextBasa(item))
#        _rwfiles.SaveFilePckle(item + "_basa", dan)
        all_dan[ind]=(item, dan)
        ind += 1
    return all_dan

def _form_dan(_rwfiles, _cd, all_basa:dict, _ncomplaint=500, _word=25):
    print(" ==>>==  Формируем данные ==<<== ")    
    _all_dan = dict()

    for key, val in all_basa.items():
        _all_dan[key] = _cd.complaint_train_test(val, _ncomplaint, _word)

    return _all_dan

def _form_dan_to_model(_rwfiles, _cd):

    z =  _rwfiles.LoadFilePckle("DanALL_text")
    _cd.LoadDanALLtext(z)

    maxWordsCount, xLen, step = 1000, 50, 10

    trein_text, test_text, tokenizer =  _cd.tokenizer_train_test(maxWordsCount)   # maxWordsCount=100
    #Формируем обучающую и тестовую выборку
    xTrain, yTrain = _cd.createSetsMultiClasses(_cd.trainWordIndexes, xLen, step) #извлекаем обучающую выборку
    xTest, yTest = _cd.createSetsMultiClasses(_cd.testWordIndexes, xLen, step)    #извлекаем тестовую выборку

    #Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
    xTrain01 = tokenizer.sequences_to_matrix(xTrain.tolist())                       # подаем xTrain в виде списка чтобы метод успешно сработал
    xTest01 = tokenizer.sequences_to_matrix(xTest.tolist())                         # подаем xTest в виде списка чтобы метод успешно сработал
    return xTrain01, yTrain, xTest01, yTest, _cd.category


def stage_x(CountBot, Countlayer, _max_mutation,_repeat, name_test, dan):
    _writeread = WriteRead.WriteReadDan()                       # класс

    _generationModels = GenDanDense.GenerationModels(CountBot)
    _generationModels.CreateModels(True, (1, Countlayer, 1))
    _models =_generationModels.Models

    for i in range(_repeat):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!  "+str(i)+ "   !!!!!!!!!!!!!!!!!!!!!!!!!")
        _fits = CFitGen.FITs(dan)
        _fits.SetModel(_models)
        _fits.run()
        history, models_max, modelsprn = _fits.ReadHistory()

        namefile = name_test + str(i)
        _writeread.Set(namefile)

        _writeread.write_file_json(modelsprn)
        _writeread.write_file_pic(history, models_max)

        _generationModels.CreateCopyModels( models_max, _max_mutation)
        _models = _generationModels.Get()

# модули для уформирования и генерации исходных данных
def SetDan():
#  функции для промежуточных вычислений
    print("--==** базовые данные сформированны  **==--")
    all_dan = _form_basa_dan_treatment(_rwfiles, _cd)
    _rwfiles.SaveFilePckle("all_basa", all_dan)

    print("--==** Считываем базу all_basa и получаем статистику  **==--")
    all_basa =  _rwfiles.LoadFilePckle("all_basa")
    _stat_dan = _cd.StatDanComplaint(all_basa)

# считаем нужное кол-во слов по каждой группе
    _cd.calc_complaint_and_work(_stat_dan, 500, 25)

    print(" Генерим данные train и test 500 обращений по 30 слов ")
    _all_dan_30word =  _form_dan(_rwfiles, _cd, all_basa, 500, 30)
    _rwfiles.SaveFilePckle("all_dan_30word", _all_dan_30word)       # сохраняем

    z =  _rwfiles.LoadFilePckle("all_dan_30word")
    _cd.Create_Train_Test(z)                                        # преобразую train_test в длинную строку
    _rwfiles.SaveFilePckle("DanALL_text", _cd.DanALL_text)          # и сохраняем
    
    z =  _rwfiles.LoadFilePckle("DanALL_text")                      # считываем сохраненную базу
    _cd.LoadDanALLtext(z)                                           # Загружаем в class - ConverDan

if __name__ == "__main__":
   
    _rwfiles = RWFiles("O:/Python/TestDima/TestDima/Болезни")
    _cd = CD()

    xTrain01, yTrain, xTest01, yTest, category=  _form_dan_to_model(_rwfiles, _cd)
    dan  = [xTrain01, yTrain, xTest01, yTest]
   
# ================
# Запускаем генетический генератор
# Тест 0: 
#   20 - кол-во ботов
#   15 - кол-во блоков в блоке:   Dense, Activation, Dropout 
#   0.5- max Dropout
#   5 - число повторений
#   T0_b20_l15_e10_ - файл json - записываются 3 лучшие результата
#    

    # запускаем 0 тест для поиско кол-во слоев
#    stage_x(20, 15, 0.5, 5, "T0_b20_l15_e10_", dan)

    # запускаем 1 тест, весь перечень активации
#    stage_x(20, 8, 0.5, 5, "T1_b20_l8_e10_", dan) 

    # запускаем 2 тест, весь перечень активации
#    stage_x(20, 5, 0.75, 5, "T2_b20_l_5_e10_", dan) 

    # запускаем 3 тест, весь перечень активации
# typeActivation = ["sigmoid", "softsign"] => "val_accuracy": "0.982083"
#"18": { проверить и тучнить переменные
#    "0": [ "Dense(1960,  Activation= sigmoid)" ],
#    "1": [ "Dense(1740,  Activation= softsign)" ],
#    "val_accuracy": "0.982083"  },

    stage_x(10, 3, 0.5, 3, "T3_b10_l_3_e10_", dan) 

# 10 -бот 3 блока Dropout 0.5 
# промежуточные данные лежат в
    i=1