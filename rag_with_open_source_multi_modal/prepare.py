import os
from tqdm import tqdm
import wikipedia
import shutil


wiki_titles = { # the key is imgs class and the value is wiki title
    'daisy': 'Bellis perennis',
    'dandelion': 'Taraxacum',
    'lotus': 'Nelumbo nucifera',
    'rose': 'Rose',
    'sunflower': 'Common sunflower',
    'tulip': 'Tulip',
    'bellflower':'Campanula'
}

imges_classes = ['magnolia',
 'california_poppy',
 'tulip',
 'calendula',
 'iris',
 'coreopsis',
 'bellflower',
 'rose',
 'dandelion',
 'common_daisy',
 'water_lily',
 'carnation',
 'black_eyed_susan',
 'daffodil',
 'sunflower',
 'astilbe']

def set_wiki_pair():
    new_path = '../data/flowers/'
    # each class has 10 images and one text file content from the wiki page
    for cls in tqdm(imges_classes):
        cls_pth = os.path.join(new_path, cls)
        # 判断当前的花名是否在wiki_titles中，不在则跳过
        if cls not in wiki_titles.keys():
            continue
        print(cls)
        # page_content = wikipedia.page(wiki_titles[cls], auto_suggest=False).content
        page_content = wikipedia.summary(wiki_titles[cls] ,auto_suggest=False)

        print(page_content)

        if not os.path.exists(cls_pth):
            print('Creating {} folder'.format(cls))
        else:
            #save the text file
            files_name= cls+'.txt'
            with open(os.path.join(cls_pth, files_name), 'w') as f:
                f.write(page_content)



if __name__ == '__main__':
    set_wiki_pair()