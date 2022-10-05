import json, os

HEAD = ['<!DOCTYPE html>','<html>', '<body>']
TAIL = ['</body>', '</html>']




class HTMLGenerator:
    def __init__(self, img_dirs, crop_size=(256, 192)):
        self.img_dirs = img_dirs
    def _generate_head(self):
        out_string = '\n'.join(HEAD) + '\n'
        return out_string

    def _generate_tail(self):
        out_string = '\n'.join(TAIL) + '\n'
        return out_string
            

    def _generate_subject_section(self, exp_name='default', epoch=0, subset='Full'):
        out_string = '<h1> Visualization </h1>' + '\n'
        #out_string += '<h2>Exp Meta</h2>' + '\n'
        #out_string += '<p>Exp Name: %s</p>' % exp_name + '\n'
        #out_string += '<p>Epoch: %d</p>' % epoch + '\n'
        #out_string += '<p>Subset: %s </p>' % subset + '\n'
        return out_string

    def _generate_retrieval_table(self, crop_size=(256, 192)):
        # head
        out_string = '<h2>Retrieval Result</h2>' + '\n'
        out_string += '<table>' + '\n'
        out_string += '<tr>' + '\n'
        out_string += '<th>index</th>'  + '\n'
        for name in self.img_dirs:
            out_string += '<th>%s</th>' % name + '\n'
        out_string += '</tr>' + '\n'
        # body
        keys = list(self.img_dirs.keys())
        N = len(os.listdir(self.img_dirs[keys[0]]))
        for i in range(1,N+1):
            out_string += '<tr>' + '\n'
            out_string += '<td>index</td>'  + '\n'
            for name in self.img_dirs:
                out_string += '<td>%s</td>' % name + '\n'
            out_string += '</tr>' + '\n'
            out_string += '<tr>' + '\n'
            out_string += '<td>%d</td>' % i + '\n'
            fn = 'generated_%d.jpg' % i
            for folder in self.img_dirs:
                img_path = os.path.join(img_dirs[folder], fn)
                #print(crop_size[1])
                out_string += '<td><img src="%s" height=256 width=%s" style="border:solid; border-color:red;"/></td>' % (img_path, crop_size[1] * 5) + '\n'
            out_string += '</tr>' + '\n'

        return out_string

   
    def generate(self, out_path='tmp.html', crop_size=(256, 192)):
        out_string = self._generate_head()
        #print(out_string)
        #subset_title = 'false_only_{} (if True, only those queries with R@10 = 0 displayed.)'
        out_string += self._generate_subject_section()
        out_string += self._generate_retrieval_table(crop_size=crop_size)
        out_string += self._generate_tail()
        with open(out_path, 'w') as f:
            f.write(out_string)


if __name__ == '__main__':
    if False:
        root = '../data/'
        img_dirs = {
            "dior": "/shared/rsaas/shaoyuc3/dior_PBAFN/checkpoints/DIOR_32/generate_latest",
            "dior_style": "/shared/rsaas/shaoyuc3/dior_PBAFN/checkpoints/dior_style/generate_latest"
        }
        Generator = HTMLGenerator(img_dirs=img_dirs)
        Generator.generate()
    else:
        import argparse
        parser = argparse.ArgumentParser(description='SSIM Eval.')
        parser.add_argument('--crop_size', type=str, default='256, 176', help='then crop to this size')
        parser.add_argument('--square', action='store_true', help='is square image. (256x256)')
        opt = parser.parse_args()
        opt.crop_size = tuple(map(int, opt.crop_size.split(', ')))
        if opt.square: #opt.crop_size >= 250:
            opt.crop_size = (opt.crop_size[0], opt.crop_size[0])
        root = '"/shared/rsaas/shaoyuc3/dior_PBAFN'
        img_dirs = {
            "Dior-ovnet from scratch test": "checkpoints/image_compare/from_scratch/unpaired/dior_warp",
            "Dior-flowstyle from scratch test": "checkpoints/image_compare/from_scratch/unpaired/dior_style",
        }
        Generator = HTMLGenerator(img_dirs=img_dirs, crop_size=opt.crop_size)
        Generator.generate("finetune_unpaired.html", crop_size=opt.crop_size)
        
        
        #python tools/generate_http.py --crop_size '256, 192'
        #python -m http.server 9999