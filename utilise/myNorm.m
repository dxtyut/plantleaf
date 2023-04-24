
function outmatr=myNorm(A,mode)


switch mode
    case 'h'
        %  normalize each row to unit
        outmatr = A./repmat(sqrt(sum(A.^2,2)),1,size(A,2));
    case 'v'
        %  normalize each column to unit
        outmatr = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1);
end