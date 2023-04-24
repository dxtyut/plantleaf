function im_resized = preprocessImage(im, keepAspect, imageSize, averageColour, scale, border) 
        
        w = size(im,2) ;
        h = size(im,1) ;
        factor = [(imageSize(1)+border(1))/h, (imageSize(2)+border(2))/w]*scale;

        if keepAspect
            factor = max(factor);
        end

        im_resized = imresize(single(im),'scale', factor, 'method', 'bilinear') ;

        w = size(im_resized,2) ;
        h = size(im_resized,1) ;

        im_resized = imcrop(im_resized, [fix((w-imageSize(1)*scale)/2)+1, ...
                    fix((h-imageSize(2)*scale)/2)+1, ...
                    round(imageSize(1)*scale)-1, round(imageSize(2)*scale)-1]);

        im_resized = bsxfun(@minus, im_resized, averageColour) ;
