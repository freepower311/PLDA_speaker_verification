# PLDA_speaker_verification

This project is comparation of hinge loss function and the logical loss function in task of speaker verification.

## Article
It is mainly based on this [article](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/cumani_icassp2011_4852.pdf). Pairwise Discriminative Speaker Verification in the I-Vector Space / Sandro Cumani, Niko Brummer, Lukas Burget, Pietro Laface, Oldrich Plchot, and Vasileios Vasilakakis. // IEEE TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 21, NO. 6, 2013. С. 1217-1227.

## Function description
### score (w, F1, F2)
Implementation of the formula (1)

<img src="https://render.githubusercontent.com/render/math?math=S=2 \Phi _e^T  \Lambda  \Phi _t %2B ((\Phi _e^T  \Gamma )  \circ \Phi _e^T)11^T %2B 11^T(\Phi _t \circ (\Gamma\Phi _t)) %2B \Phi _e^Tc1^T %2B 1c^T \Phi _t %2B k11^T"/> &nbsp;&nbsp;&nbsp;&nbsp; (1)

<img src="https://render.githubusercontent.com/render/math?math=\circ"/> denotes the Hadamard product

1 denotes vector of ones

where w is a vector of classifier parameters, F1 and F2 are matrices Φ_e and Φ_t containing sets of i-vectors, returns a matrix of similarity estimates between Φ_e and Φ_t by columns.

### sigmoid(x)
a numerically stable sigmoid function, where x is any vector. Returns σ(x).

### calc_metrics(targets_scores, imposter_scores)
Calculating classification quality metrics, returns FRR, FAR, and EER. Takes as input vectors containing similarity scores obtained by the classifier, where targets_scores includes those for which y = +1, and imposter_scores, respectively, for y = -1.

### risk_logistic (w, F, Y, lmb)
implementation of formulas (2), (3), (4) and (5)

<img src="https://render.githubusercontent.com/render/math?math=E(w)= \sum_{n=1}^N  \alpha_n E_{LR} (y_n s_n) %2B  \frac{\lambda}{2}  \| w \| ^2"/>, 
 &nbsp;&nbsp;&nbsp;&nbsp; (2)

<img src="https://render.githubusercontent.com/render/math?math=E_{LR}(ys)= log(1 %2B exp(-ys))"/>, &nbsp;&nbsp;&nbsp;&nbsp; (3)

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E(y_ns_n)}{\partial s_n} = -y \sigma (-ys)"/>, &nbsp;&nbsp;&nbsp;&nbsp; (4)

<img src="https://render.githubusercontent.com/render/math?math=\nabla E(w) =  \begin{bmatrix}\nabla_ \Lambda L   \\ \nabla_  \Gamma  L  \\ \nabla_c  L \\ \nabla_k L \end{bmatrix} ="/> <img src="https://render.githubusercontent.com/render/math?math=\begin{bmatrix} 2  * vec( \Phi G  \Phi ^T)   \\ 2  * vec( \Phi [\Phi ^T  \circ (G11^T)])  \\ 2  * 1^T[ \Phi ^t \circ (G11^T)] \\ 1^T G1 \end{bmatrix} %2B  \lambda w"/> &nbsp;&nbsp;&nbsp;&nbsp; (5)

where w is the vector of the classifier parameters, F is the matrix Φ containing the training set of i-vectors, Y is the response matrix, lmb is the regularization coefficient λ. y_ij is either +1 or -1 if the i-th and j-th columns of the matrix Φ belong to one or different people, respectively. The function returns the value and gradient of the logistic regression loss function.

### risk_hinge (w, F, Y, lmb)
Implementation of formulas (2), (6), (7) and (5)

<img src="https://render.githubusercontent.com/render/math?math=E_{LR}(ys) = max(0, 1- ys)"/> &nbsp;&nbsp;&nbsp;&nbsp; (6)

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial E(y_ns_n)}{\partial s_n} = \begin{cases}0 ,ys  \geq  1\\-y, otherwise \end{cases}"/> &nbsp;&nbsp;&nbsp;&nbsp; (7)

where w is the vector of the classifier parameters, F is the matrix Φ containing the training set of i-vectors, Y is the response matrix, and lmb is the regularization coefficient λ. The function returns the value and gradient of the hinge loss function.
