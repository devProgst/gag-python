from gag_python.gag import GAGModel


test_images = [
    [ 'Oldman', './test_images/old.jpg' ],
    [ 'Middle', './test_images/middle.jpg' ],
    [ 'Child', './test_images/child.jpg' ]
]

gag = GAGModel()

for mtype, img in test_images:
    rez = gag.get_age_gender(img, returnJson=True)
    print(mtype, 'is', rez[0], 'and', rez[1])