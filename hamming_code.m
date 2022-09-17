close all;clear all;clc;
SNRdB=1:1:11;                            %SNR in dB
SNR=10.^(SNRdB./10);                     %SNR in linear scale   
info_word_length=1000000;                 %No. of information words   
n=7;k=4;                                 %Parameters of hamming code   
ber=zeros(length(SNR),2);                %Simulated BER   
info_word=floor(2*rand(k,info_word_length));    %Generation of 0 and 1 for infromation bits
code_bit5=xor(info_word(1,:),xor(info_word(2,:),info_word(3,:)));   %First Parity Bit
code_bit6=xor(info_word(1,:),xor(info_word(3,:),info_word(4,:)));   %Second Parity Bit
code_bit7=xor(info_word(1,:),xor(info_word(2,:),info_word(4,:)));   %Third Parity Bit
code_word=[info_word;code_bit5;code_bit6;code_bit7];       %Coded information Word with parity bits
code_word(code_word==0)=-1;              %Converting 0 bits to 1
decoded_bit=zeros(n,info_word_length);             %HARD Decoding Output   
decoded_block=zeros(n,info_word_length);           %SOFT Decoding Output
H=[1 1 1;1 0 1;1 1 0;0 1 1;1 0 0;0 1 0;0 0 1];     %Parity Check Matrix
C=de2bi((0:2^(k)-1));                       %All bits of length k(Stored in valid code words matrix 'C')
C(1:16,5)=xor(C(:,1),xor(C(:,2),C(:,3)));   %First Parity Bit
C(1:16,6)=xor(C(:,1),xor(C(:,3),C(:,4)));   %Second Parity Bit
C(1:16,7)=xor(C(:,1),xor(C(:,2),C(:,4)));   %Third Parity Bit
distance=zeros(1,2^k);
for i=1:length(SNR)
    y=(sqrt(SNR(i))*code_word)+randn(n,info_word_length);     %Received Codes
    
    %For BIT(Hard) Detection
    decoded_bit(y>0)=1;                  %All positive received bits converted to +1
    decoded_bit(y<0)=0;                   %All negative received bits converted to 0
    
    %Decoding Received Codes into valid codewords
    for l=1:info_word_length
        %HARD Decoding
        hi= decoded_bit(:,l)'*H;          %Syndrome Detection
        for j=1:n               %Matching 'hi' to every row vector of H and flipping the corresponding bit of 'z' using xor
            if (hi==H(j,:))
                decoded_bit(j,l)=~decoded_bit(j,l);    %NOT operation on the corresponding bit
            end
        end
        %SOFT Decoding
        for m=1:(k^2)           %Tacking distance of each column of the received word to a valid codeword
            distance(m)=norm(y(:,l)-C(m,:)');
        end
        [minval,minind]=min(distance);       %Finding index of the minimum distance valid codeword
        
        decoded_block(:,l)=C(minind,:);      %Decoding as the min distance codewor 
    end
    ber(i,1)=length(find(decoded_bit(1:4,:)~=info_word));     %BER in BIT Detection
    ber(i,2)=length(find(decoded_block(1:4,:)~=info_word));   %BER in BLOCK Detection
end
ber=ber/(k*info_word_length);
semilogy(SNRdB,ber(:,1),'r-<','linewidth',2.0)    %Simulated BER in HARD decoding
hold on
semilogy(SNRdB,ber(:,2),'m-<','linewidth',2.0)    %Simulated BER in SOFT Decoding
hold on
p=qfunc(sqrt(SNR));
BER_HARD=zeros(1,length(SNR));
for j=2:n
    BER_HARD=BER_HARD+nchoosek(n,j).*(p.^j).*((1-p).^(n-j));
end
semilogy(SNRdB,BER_HARD,'k-','linewidth',2.0);         %Theroritical BER in HARD decoding
hold on
BER_SOFT=(7.*qfunc(sqrt(3*SNR)))+(7.*qfunc(sqrt(4*SNR)))+(qfunc(sqrt(7*SNR)));
semilogy(SNRdB,BER_SOFT,'b-','linewidth',2.0)          %Theoritical BER in SOFT decoding
title('BIT and BLOCK Detection for (7,4) Hamming Code');xlabel('SNR(dB)');ylabel('BER');
legend('Simulated BER(Hard Decoding)','Simulated BER(Soft Decoding)','Theoritical BER(Hard DEcoding)','Theoritical BER(Soft Decoding)');
axis tight
grid