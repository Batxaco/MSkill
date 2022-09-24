from utils.MlSkillsOne import TextAnalizer
from utils.MlSkillsTwo import Predictor
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

def main():

 #   to_trantor1 = TextAnalizer()
    to_trantor2 = Predictor()

    list_list_elements = to_trantor2.flat_instance['element'].map(lambda row: [row]).tolist()

    model = Word2Vec(list_list_elements, min_count=1, workers=4)

    print(model.wv.index_to_key)

    print("BYE")


if __name__ == '__main__':
    main()

