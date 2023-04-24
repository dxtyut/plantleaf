
function outname = abbrName(inputname)

    if strcmp(inputname,'imagenet-vgg-verydeep-16')
        outname = 'VGG16';
    end

    if strcmp(inputname,'imagenet-vgg-verydeep-19')
        outname = 'VGG19';
    end

    if strcmp(inputname,'imagenet-resnet-50-dag')
        outname = 'Res';
    end