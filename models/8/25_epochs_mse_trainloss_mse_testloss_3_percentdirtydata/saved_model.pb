??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
|
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_48/kernel
u
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel* 
_output_shapes
:
??*
dtype0
s
dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_48/bias
l
!dense_48/bias/Read/ReadVariableOpReadVariableOpdense_48/bias*
_output_shapes	
:?*
dtype0
{
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_49/kernel
t
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel*
_output_shapes
:	?@*
dtype0
r
dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_49/bias
k
!dense_49/bias/Read/ReadVariableOpReadVariableOpdense_49/bias*
_output_shapes
:@*
dtype0
z
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_50/kernel
s
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes

:@ *
dtype0
r
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_50/bias
k
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes
: *
dtype0
z
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_51/kernel
s
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel*
_output_shapes

: @*
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
:@*
dtype0
{
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?* 
shared_namedense_52/kernel
t
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
_output_shapes
:	@?*
dtype0
s
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_52/bias
l
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes	
:?*
dtype0
|
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_53/kernel
u
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel* 
_output_shapes
:
??*
dtype0
s
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_53/bias
l
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_48/kernel/m
?
*Adam/dense_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_48/bias/m
z
(Adam/dense_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_49/kernel/m
?
*Adam/dense_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_49/bias/m
y
(Adam/dense_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_50/kernel/m
?
*Adam/dense_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/dense_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_50/bias/m
y
(Adam/dense_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_51/kernel/m
?
*Adam/dense_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/m*
_output_shapes

: @*
dtype0
?
Adam/dense_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_51/bias/m
y
(Adam/dense_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*'
shared_nameAdam/dense_52/kernel/m
?
*Adam/dense_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/m*
_output_shapes
:	@?*
dtype0
?
Adam/dense_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_52/bias/m
z
(Adam/dense_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_53/kernel/m
?
*Adam/dense_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_53/bias/m
z
(Adam/dense_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_48/kernel/v
?
*Adam/dense_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_48/bias/v
z
(Adam/dense_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_49/kernel/v
?
*Adam/dense_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_49/bias/v
y
(Adam/dense_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_50/kernel/v
?
*Adam/dense_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/dense_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_50/bias/v
y
(Adam/dense_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_nameAdam/dense_51/kernel/v
?
*Adam/dense_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/v*
_output_shapes

: @*
dtype0
?
Adam/dense_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_51/bias/v
y
(Adam/dense_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*'
shared_nameAdam/dense_52/kernel/v
?
*Adam/dense_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/v*
_output_shapes
:	@?*
dtype0
?
Adam/dense_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_52/bias/v
z
(Adam/dense_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_53/kernel/v
?
*Adam/dense_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_53/bias/v
z
(Adam/dense_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?X
value?XB?X B?X
?
encoder
decoder
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
 iter

!beta_1

"beta_2
	#decay
$learning_rate%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?*
Z
%0
&1
'2
(3
)4
*5
+6
,7
-8
.9
/10
011*
Z
%0
&1
'2
(3
)4
*5
+6
,7
-8
.9
/10
011*
* 
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

6serving_default* 
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
?

%kernel
&bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
?

'kernel
(bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*
?

)kernel
*bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses*
.
%0
&1
'2
(3
)4
*5*
.
%0
&1
'2
(3
)4
*5*
* 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?

+kernel
,bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
?

-kernel
.bias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses*
?

/kernel
0bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses*
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
.
+0
,1
-2
.3
/4
05*
.
+0
,1
-2
.3
/4
05*
* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_48/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_48/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_49/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_49/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_50/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_50/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_51/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_51/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_52/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_52/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_53/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_53/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

q0
r1*
* 
* 
* 
* 
* 
* 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 
* 
* 

%0
&1*

%0
&1*
* 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 

'0
(1*

'0
(1*
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 

)0
*1*

)0
*1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
* 
* 
* 
 
0
1
2
3*
* 
* 
* 

+0
,1*

+0
,1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 

-0
.1*

-0
.1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 

/0
01*

/0
01*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 
* 
* 
* 
 
0
1
2
3*
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
rl
VARIABLE_VALUEAdam/dense_48/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_48/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_49/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_49/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_50/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_50/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_51/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_51/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_52/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_52/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_53/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_53/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_48/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_48/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_49/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_49/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_50/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_50/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_51/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_51/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_52/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_52/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_53/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_53/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_48/kerneldense_48/biasdense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1456311
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_48/kernel/Read/ReadVariableOp!dense_48/bias/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp!dense_49/bias/Read/ReadVariableOp#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_48/kernel/m/Read/ReadVariableOp(Adam/dense_48/bias/m/Read/ReadVariableOp*Adam/dense_49/kernel/m/Read/ReadVariableOp(Adam/dense_49/bias/m/Read/ReadVariableOp*Adam/dense_50/kernel/m/Read/ReadVariableOp(Adam/dense_50/bias/m/Read/ReadVariableOp*Adam/dense_51/kernel/m/Read/ReadVariableOp(Adam/dense_51/bias/m/Read/ReadVariableOp*Adam/dense_52/kernel/m/Read/ReadVariableOp(Adam/dense_52/bias/m/Read/ReadVariableOp*Adam/dense_53/kernel/m/Read/ReadVariableOp(Adam/dense_53/bias/m/Read/ReadVariableOp*Adam/dense_48/kernel/v/Read/ReadVariableOp(Adam/dense_48/bias/v/Read/ReadVariableOp*Adam/dense_49/kernel/v/Read/ReadVariableOp(Adam/dense_49/bias/v/Read/ReadVariableOp*Adam/dense_50/kernel/v/Read/ReadVariableOp(Adam/dense_50/bias/v/Read/ReadVariableOp*Adam/dense_51/kernel/v/Read/ReadVariableOp(Adam/dense_51/bias/v/Read/ReadVariableOp*Adam/dense_52/kernel/v/Read/ReadVariableOp(Adam/dense_52/bias/v/Read/ReadVariableOp*Adam/dense_53/kernel/v/Read/ReadVariableOp(Adam/dense_53/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1456808
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_48/kerneldense_48/biasdense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/biastotalcounttotal_1count_1Adam/dense_48/kernel/mAdam/dense_48/bias/mAdam/dense_49/kernel/mAdam/dense_49/bias/mAdam/dense_50/kernel/mAdam/dense_50/bias/mAdam/dense_51/kernel/mAdam/dense_51/bias/mAdam/dense_52/kernel/mAdam/dense_52/bias/mAdam/dense_53/kernel/mAdam/dense_53/bias/mAdam/dense_48/kernel/vAdam/dense_48/bias/vAdam/dense_49/kernel/vAdam/dense_49/bias/vAdam/dense_50/kernel/vAdam/dense_50/bias/vAdam/dense_51/kernel/vAdam/dense_51/bias/vAdam/dense_52/kernel/vAdam/dense_52/bias/vAdam/dense_53/kernel/vAdam/dense_53/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1456953??	
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455466

inputs$
dense_48_1455426:
??
dense_48_1455428:	?#
dense_49_1455443:	?@
dense_49_1455445:@"
dense_50_1455460:@ 
dense_50_1455462: 
identity?? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1455412?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_48_1455426dense_48_1455428*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_1455425?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1455443dense_49_1455445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_1455442?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_1455460dense_50_1455462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_1455459x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_16_layer_call_fn_1456345

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_49_layer_call_fn_1456541

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_1455442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1456399

inputs;
'dense_48_matmul_readvariableop_resource:
??7
(dense_48_biasadd_readvariableop_resource:	?:
'dense_49_matmul_readvariableop_resource:	?@6
(dense_49_biasadd_readvariableop_resource:@9
'dense_50_matmul_readvariableop_resource:@ 6
(dense_50_biasadd_readvariableop_resource: 
identity??dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/BiasAdd/ReadVariableOp?dense_49/MatMul/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?dense_50/MatMul/ReadVariableOp`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  q
flatten_8/ReshapeReshapeinputsflatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_48/MatMulMatMulflatten_8/Reshape:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_50/MatMulMatMuldense_49/Relu:activations:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? b
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*'
_output_shapes
:????????? j
IdentityIdentitydense_50/Relu:activations:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
/__inference_autoencoder_8_layer_call_fn_1456137
x
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1455898s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
/__inference_sequential_17_layer_call_fn_1455717
dense_51_input
unknown: @
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_51_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455702s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:????????? 
(
_user_specified_namedense_51_input
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455844
dense_51_input"
dense_51_1455827: @
dense_51_1455829:@#
dense_52_1455832:	@?
dense_52_1455834:	?$
dense_53_1455837:
??
dense_53_1455839:	?
identity?? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCalldense_51_inputdense_51_1455827dense_51_1455829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_1455646?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_1455832dense_52_1455834*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_1455663?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_1455837dense_53_1455839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_1455680?
reshape_8/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_1455699u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:W S
'
_output_shapes
:????????? 
(
_user_specified_namedense_51_input
?

?
E__inference_dense_50_layer_call_and_return_conditional_losses_1455459

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
/__inference_sequential_16_layer_call_fn_1455481
flatten_8_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455466o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455792

inputs"
dense_51_1455775: @
dense_51_1455777:@#
dense_52_1455780:	@?
dense_52_1455782:	?$
dense_53_1455785:
??
dense_53_1455787:	?
identity?? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCallinputsdense_51_1455775dense_51_1455777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_1455646?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_1455780dense_52_1455782*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_1455663?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_1455785dense_53_1455787*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_1455680?
reshape_8/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_1455699u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455608
flatten_8_input$
dense_48_1455592:
??
dense_48_1455594:	?#
dense_49_1455597:	?@
dense_49_1455599:@"
dense_50_1455602:@ 
dense_50_1455604: 
identity?? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallflatten_8_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1455412?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_48_1455592dense_48_1455594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_1455425?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1455597dense_49_1455599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_1455442?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_1455602dense_50_1455604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_1455459x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1455898
x)
sequential_16_1455871:
??$
sequential_16_1455873:	?(
sequential_16_1455875:	?@#
sequential_16_1455877:@'
sequential_16_1455879:@ #
sequential_16_1455881: '
sequential_17_1455884: @#
sequential_17_1455886:@(
sequential_17_1455888:	@?$
sequential_17_1455890:	?)
sequential_17_1455892:
??$
sequential_17_1455894:	?
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallxsequential_16_1455871sequential_16_1455873sequential_16_1455875sequential_16_1455877sequential_16_1455879sequential_16_1455881*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455466?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_1455884sequential_17_1455886sequential_17_1455888sequential_17_1455890sequential_17_1455892sequential_17_1455894*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455702?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456102
input_1)
sequential_16_1456075:
??$
sequential_16_1456077:	?(
sequential_16_1456079:	?@#
sequential_16_1456081:@'
sequential_16_1456083:@ #
sequential_16_1456085: '
sequential_17_1456088: @#
sequential_17_1456090:@(
sequential_17_1456092:	@?$
sequential_17_1456094:	?)
sequential_17_1456096:
??$
sequential_17_1456098:	?
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_1456075sequential_16_1456077sequential_16_1456079sequential_16_1456081sequential_16_1456083sequential_16_1456085*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455556?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_1456088sequential_17_1456090sequential_17_1456092sequential_17_1456094sequential_17_1456096sequential_17_1456098*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455792?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455702

inputs"
dense_51_1455647: @
dense_51_1455649:@#
dense_52_1455664:	@?
dense_52_1455666:	?$
dense_53_1455681:
??
dense_53_1455683:	?
identity?? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCallinputsdense_51_1455647dense_51_1455649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_1455646?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_1455664dense_52_1455666*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_1455663?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_1455681dense_53_1455683*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_1455680?
reshape_8/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_1455699u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1455986
x)
sequential_16_1455959:
??$
sequential_16_1455961:	?(
sequential_16_1455963:	?@#
sequential_16_1455965:@'
sequential_16_1455967:@ #
sequential_16_1455969: '
sequential_17_1455972: @#
sequential_17_1455974:@(
sequential_17_1455976:	@?$
sequential_17_1455978:	?)
sequential_17_1455980:
??$
sequential_17_1455982:	?
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallxsequential_16_1455959sequential_16_1455961sequential_16_1455963sequential_16_1455965sequential_16_1455967sequential_16_1455969*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455556?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_1455972sequential_17_1455974sequential_17_1455976sequential_17_1455978sequential_17_1455980sequential_17_1455982*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455792?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?

b
F__inference_reshape_8_layer_call_and_return_conditional_losses_1455699

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_16_layer_call_fn_1456328

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455466o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_48_layer_call_and_return_conditional_losses_1455425

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_17_layer_call_fn_1455824
dense_51_input
unknown: @
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_51_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455792s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:????????? 
(
_user_specified_namedense_51_input
?

?
E__inference_dense_52_layer_call_and_return_conditional_losses_1456612

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455864
dense_51_input"
dense_51_1455847: @
dense_51_1455849:@#
dense_52_1455852:	@?
dense_52_1455854:	?$
dense_53_1455857:
??
dense_53_1455859:	?
identity?? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCalldense_51_inputdense_51_1455847dense_51_1455849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_1455646?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_1455852dense_52_1455854*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_1455663?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_1455857dense_53_1455859*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_1455680?
reshape_8/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_1455699u
IdentityIdentity"reshape_8/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:W S
'
_output_shapes
:????????? 
(
_user_specified_namedense_51_input
?

?
/__inference_autoencoder_8_layer_call_fn_1456166
x
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1455986s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
*__inference_dense_53_layer_call_fn_1456621

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_1455680p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455628
flatten_8_input$
dense_48_1455612:
??
dense_48_1455614:	?#
dense_49_1455617:	?@
dense_49_1455619:@"
dense_50_1455622:@ 
dense_50_1455624: 
identity?? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallflatten_8_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1455412?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_48_1455612dense_48_1455614*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_1455425?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1455617dense_49_1455619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_1455442?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_1455622dense_50_1455624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_1455459x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?

?
E__inference_dense_51_layer_call_and_return_conditional_losses_1456592

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_1455412

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_49_layer_call_and_return_conditional_losses_1455442

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_17_layer_call_fn_1456416

inputs
unknown: @
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455702s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_8_layer_call_fn_1455925
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1455898s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_dense_48_layer_call_fn_1456521

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_1455425p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1456467

inputs9
'dense_51_matmul_readvariableop_resource: @6
(dense_51_biasadd_readvariableop_resource:@:
'dense_52_matmul_readvariableop_resource:	@?7
(dense_52_biasadd_readvariableop_resource:	?;
'dense_53_matmul_readvariableop_resource:
??7
(dense_53_biasadd_readvariableop_resource:	?
identity??dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0{
dense_51/MatMulMatMulinputs&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
dense_52/MatMulMatMuldense_51/Relu:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_53/MatMulMatMuldense_52/Relu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????i
dense_53/SigmoidSigmoiddense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????S
reshape_8/ShapeShapedense_53/Sigmoid:y:0*
T0*
_output_shapes
:g
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape_8/ReshapeReshapedense_53/Sigmoid:y:0 reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????m
IdentityIdentityreshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_dense_50_layer_call_and_return_conditional_losses_1456572

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_dense_52_layer_call_and_return_conditional_losses_1455663

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_51_layer_call_fn_1456581

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_1455646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_flatten_8_layer_call_fn_1456506

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1455412a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456223
xI
5sequential_16_dense_48_matmul_readvariableop_resource:
??E
6sequential_16_dense_48_biasadd_readvariableop_resource:	?H
5sequential_16_dense_49_matmul_readvariableop_resource:	?@D
6sequential_16_dense_49_biasadd_readvariableop_resource:@G
5sequential_16_dense_50_matmul_readvariableop_resource:@ D
6sequential_16_dense_50_biasadd_readvariableop_resource: G
5sequential_17_dense_51_matmul_readvariableop_resource: @D
6sequential_17_dense_51_biasadd_readvariableop_resource:@H
5sequential_17_dense_52_matmul_readvariableop_resource:	@?E
6sequential_17_dense_52_biasadd_readvariableop_resource:	?I
5sequential_17_dense_53_matmul_readvariableop_resource:
??E
6sequential_17_dense_53_biasadd_readvariableop_resource:	?
identity??-sequential_16/dense_48/BiasAdd/ReadVariableOp?,sequential_16/dense_48/MatMul/ReadVariableOp?-sequential_16/dense_49/BiasAdd/ReadVariableOp?,sequential_16/dense_49/MatMul/ReadVariableOp?-sequential_16/dense_50/BiasAdd/ReadVariableOp?,sequential_16/dense_50/MatMul/ReadVariableOp?-sequential_17/dense_51/BiasAdd/ReadVariableOp?,sequential_17/dense_51/MatMul/ReadVariableOp?-sequential_17/dense_52/BiasAdd/ReadVariableOp?,sequential_17/dense_52/MatMul/ReadVariableOp?-sequential_17/dense_53/BiasAdd/ReadVariableOp?,sequential_17/dense_53/MatMul/ReadVariableOpn
sequential_16/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
sequential_16/flatten_8/ReshapeReshapex&sequential_16/flatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
,sequential_16/dense_48/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_48_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_16/dense_48/MatMulMatMul(sequential_16/flatten_8/Reshape:output:04sequential_16/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_16/dense_48/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_16/dense_48/BiasAddBiasAdd'sequential_16/dense_48/MatMul:product:05sequential_16/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_16/dense_48/ReluRelu'sequential_16/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_16/dense_49/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_49_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_16/dense_49/MatMulMatMul)sequential_16/dense_48/Relu:activations:04sequential_16/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
-sequential_16/dense_49/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_16/dense_49/BiasAddBiasAdd'sequential_16/dense_49/MatMul:product:05sequential_16/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
sequential_16/dense_49/ReluRelu'sequential_16/dense_49/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_16/dense_50/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_50_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
sequential_16/dense_50/MatMulMatMul)sequential_16/dense_49/Relu:activations:04sequential_16/dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
-sequential_16/dense_50/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_16/dense_50/BiasAddBiasAdd'sequential_16/dense_50/MatMul:product:05sequential_16/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
sequential_16/dense_50/ReluRelu'sequential_16/dense_50/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
,sequential_17/dense_51/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_51_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
sequential_17/dense_51/MatMulMatMul)sequential_16/dense_50/Relu:activations:04sequential_17/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
-sequential_17/dense_51/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_17/dense_51/BiasAddBiasAdd'sequential_17/dense_51/MatMul:product:05sequential_17/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
sequential_17/dense_51/ReluRelu'sequential_17/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_17/dense_52/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_52_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
sequential_17/dense_52/MatMulMatMul)sequential_17/dense_51/Relu:activations:04sequential_17/dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_17/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_17/dense_52/BiasAddBiasAdd'sequential_17/dense_52/MatMul:product:05sequential_17/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_17/dense_52/ReluRelu'sequential_17/dense_52/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_17/dense_53/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_53_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_17/dense_53/MatMulMatMul)sequential_17/dense_52/Relu:activations:04sequential_17/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_17/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_17/dense_53/BiasAddBiasAdd'sequential_17/dense_53/MatMul:product:05sequential_17/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_17/dense_53/SigmoidSigmoid'sequential_17/dense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
sequential_17/reshape_8/ShapeShape"sequential_17/dense_53/Sigmoid:y:0*
T0*
_output_shapes
:u
+sequential_17/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_17/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_17/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_17/reshape_8/strided_sliceStridedSlice&sequential_17/reshape_8/Shape:output:04sequential_17/reshape_8/strided_slice/stack:output:06sequential_17/reshape_8/strided_slice/stack_1:output:06sequential_17/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_17/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_17/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
%sequential_17/reshape_8/Reshape/shapePack.sequential_17/reshape_8/strided_slice:output:00sequential_17/reshape_8/Reshape/shape/1:output:00sequential_17/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
sequential_17/reshape_8/ReshapeReshape"sequential_17/dense_53/Sigmoid:y:0.sequential_17/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????{
IdentityIdentity(sequential_17/reshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp.^sequential_16/dense_48/BiasAdd/ReadVariableOp-^sequential_16/dense_48/MatMul/ReadVariableOp.^sequential_16/dense_49/BiasAdd/ReadVariableOp-^sequential_16/dense_49/MatMul/ReadVariableOp.^sequential_16/dense_50/BiasAdd/ReadVariableOp-^sequential_16/dense_50/MatMul/ReadVariableOp.^sequential_17/dense_51/BiasAdd/ReadVariableOp-^sequential_17/dense_51/MatMul/ReadVariableOp.^sequential_17/dense_52/BiasAdd/ReadVariableOp-^sequential_17/dense_52/MatMul/ReadVariableOp.^sequential_17/dense_53/BiasAdd/ReadVariableOp-^sequential_17/dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2^
-sequential_16/dense_48/BiasAdd/ReadVariableOp-sequential_16/dense_48/BiasAdd/ReadVariableOp2\
,sequential_16/dense_48/MatMul/ReadVariableOp,sequential_16/dense_48/MatMul/ReadVariableOp2^
-sequential_16/dense_49/BiasAdd/ReadVariableOp-sequential_16/dense_49/BiasAdd/ReadVariableOp2\
,sequential_16/dense_49/MatMul/ReadVariableOp,sequential_16/dense_49/MatMul/ReadVariableOp2^
-sequential_16/dense_50/BiasAdd/ReadVariableOp-sequential_16/dense_50/BiasAdd/ReadVariableOp2\
,sequential_16/dense_50/MatMul/ReadVariableOp,sequential_16/dense_50/MatMul/ReadVariableOp2^
-sequential_17/dense_51/BiasAdd/ReadVariableOp-sequential_17/dense_51/BiasAdd/ReadVariableOp2\
,sequential_17/dense_51/MatMul/ReadVariableOp,sequential_17/dense_51/MatMul/ReadVariableOp2^
-sequential_17/dense_52/BiasAdd/ReadVariableOp-sequential_17/dense_52/BiasAdd/ReadVariableOp2\
,sequential_17/dense_52/MatMul/ReadVariableOp,sequential_17/dense_52/MatMul/ReadVariableOp2^
-sequential_17/dense_53/BiasAdd/ReadVariableOp-sequential_17/dense_53/BiasAdd/ReadVariableOp2\
,sequential_17/dense_53/MatMul/ReadVariableOp,sequential_17/dense_53/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?

?
E__inference_dense_53_layer_call_and_return_conditional_losses_1456632

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?b
?
"__inference__wrapped_model_1455399
input_1W
Cautoencoder_8_sequential_16_dense_48_matmul_readvariableop_resource:
??S
Dautoencoder_8_sequential_16_dense_48_biasadd_readvariableop_resource:	?V
Cautoencoder_8_sequential_16_dense_49_matmul_readvariableop_resource:	?@R
Dautoencoder_8_sequential_16_dense_49_biasadd_readvariableop_resource:@U
Cautoencoder_8_sequential_16_dense_50_matmul_readvariableop_resource:@ R
Dautoencoder_8_sequential_16_dense_50_biasadd_readvariableop_resource: U
Cautoencoder_8_sequential_17_dense_51_matmul_readvariableop_resource: @R
Dautoencoder_8_sequential_17_dense_51_biasadd_readvariableop_resource:@V
Cautoencoder_8_sequential_17_dense_52_matmul_readvariableop_resource:	@?S
Dautoencoder_8_sequential_17_dense_52_biasadd_readvariableop_resource:	?W
Cautoencoder_8_sequential_17_dense_53_matmul_readvariableop_resource:
??S
Dautoencoder_8_sequential_17_dense_53_biasadd_readvariableop_resource:	?
identity??;autoencoder_8/sequential_16/dense_48/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_48/MatMul/ReadVariableOp?;autoencoder_8/sequential_16/dense_49/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_49/MatMul/ReadVariableOp?;autoencoder_8/sequential_16/dense_50/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_50/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_51/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_51/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_52/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_52/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_53/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_53/MatMul/ReadVariableOp|
+autoencoder_8/sequential_16/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
-autoencoder_8/sequential_16/flatten_8/ReshapeReshapeinput_14autoencoder_8/sequential_16/flatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
:autoencoder_8/sequential_16/dense_48/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_48_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+autoencoder_8/sequential_16/dense_48/MatMulMatMul6autoencoder_8/sequential_16/flatten_8/Reshape:output:0Bautoencoder_8/sequential_16/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
;autoencoder_8/sequential_16/dense_48/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,autoencoder_8/sequential_16/dense_48/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_48/MatMul:product:0Cautoencoder_8/sequential_16/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)autoencoder_8/sequential_16/dense_48/ReluRelu5autoencoder_8/sequential_16/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
:autoencoder_8/sequential_16/dense_49/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_49_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
+autoencoder_8/sequential_16/dense_49/MatMulMatMul7autoencoder_8/sequential_16/dense_48/Relu:activations:0Bautoencoder_8/sequential_16/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
;autoencoder_8/sequential_16/dense_49/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
,autoencoder_8/sequential_16/dense_49/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_49/MatMul:product:0Cautoencoder_8/sequential_16/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)autoencoder_8/sequential_16/dense_49/ReluRelu5autoencoder_8/sequential_16/dense_49/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
:autoencoder_8/sequential_16/dense_50/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_50_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
+autoencoder_8/sequential_16/dense_50/MatMulMatMul7autoencoder_8/sequential_16/dense_49/Relu:activations:0Bautoencoder_8/sequential_16/dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
;autoencoder_8/sequential_16/dense_50/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
,autoencoder_8/sequential_16/dense_50/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_50/MatMul:product:0Cautoencoder_8/sequential_16/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
)autoencoder_8/sequential_16/dense_50/ReluRelu5autoencoder_8/sequential_16/dense_50/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
:autoencoder_8/sequential_17/dense_51/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_51_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
+autoencoder_8/sequential_17/dense_51/MatMulMatMul7autoencoder_8/sequential_16/dense_50/Relu:activations:0Bautoencoder_8/sequential_17/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
;autoencoder_8/sequential_17/dense_51/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
,autoencoder_8/sequential_17/dense_51/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_51/MatMul:product:0Cautoencoder_8/sequential_17/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)autoencoder_8/sequential_17/dense_51/ReluRelu5autoencoder_8/sequential_17/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
:autoencoder_8/sequential_17/dense_52/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_52_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
+autoencoder_8/sequential_17/dense_52/MatMulMatMul7autoencoder_8/sequential_17/dense_51/Relu:activations:0Bautoencoder_8/sequential_17/dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
;autoencoder_8/sequential_17/dense_52/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,autoencoder_8/sequential_17/dense_52/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_52/MatMul:product:0Cautoencoder_8/sequential_17/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)autoencoder_8/sequential_17/dense_52/ReluRelu5autoencoder_8/sequential_17/dense_52/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
:autoencoder_8/sequential_17/dense_53/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_53_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+autoencoder_8/sequential_17/dense_53/MatMulMatMul7autoencoder_8/sequential_17/dense_52/Relu:activations:0Bautoencoder_8/sequential_17/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
;autoencoder_8/sequential_17/dense_53/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,autoencoder_8/sequential_17/dense_53/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_53/MatMul:product:0Cautoencoder_8/sequential_17/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,autoencoder_8/sequential_17/dense_53/SigmoidSigmoid5autoencoder_8/sequential_17/dense_53/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+autoencoder_8/sequential_17/reshape_8/ShapeShape0autoencoder_8/sequential_17/dense_53/Sigmoid:y:0*
T0*
_output_shapes
:?
9autoencoder_8/sequential_17/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;autoencoder_8/sequential_17/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;autoencoder_8/sequential_17/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3autoencoder_8/sequential_17/reshape_8/strided_sliceStridedSlice4autoencoder_8/sequential_17/reshape_8/Shape:output:0Bautoencoder_8/sequential_17/reshape_8/strided_slice/stack:output:0Dautoencoder_8/sequential_17/reshape_8/strided_slice/stack_1:output:0Dautoencoder_8/sequential_17/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5autoencoder_8/sequential_17/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :w
5autoencoder_8/sequential_17/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
3autoencoder_8/sequential_17/reshape_8/Reshape/shapePack<autoencoder_8/sequential_17/reshape_8/strided_slice:output:0>autoencoder_8/sequential_17/reshape_8/Reshape/shape/1:output:0>autoencoder_8/sequential_17/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
-autoencoder_8/sequential_17/reshape_8/ReshapeReshape0autoencoder_8/sequential_17/dense_53/Sigmoid:y:0<autoencoder_8/sequential_17/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
IdentityIdentity6autoencoder_8/sequential_17/reshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp<^autoencoder_8/sequential_16/dense_48/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_48/MatMul/ReadVariableOp<^autoencoder_8/sequential_16/dense_49/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_49/MatMul/ReadVariableOp<^autoencoder_8/sequential_16/dense_50/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_50/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_51/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_51/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_52/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_52/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_53/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2z
;autoencoder_8/sequential_16/dense_48/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_48/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_48/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_48/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_16/dense_49/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_49/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_49/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_49/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_16/dense_50/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_50/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_50/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_50/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_51/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_51/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_51/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_51/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_52/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_52/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_52/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_52/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_53/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_53/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_53/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_53/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_dense_52_layer_call_fn_1456601

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_1455663p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_50_layer_call_fn_1456561

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_1455459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?V
?
 __inference__traced_save_1456808
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_48_kernel_read_readvariableop,
(savev2_dense_48_bias_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop,
(savev2_dense_49_bias_read_readvariableop.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_48_kernel_m_read_readvariableop3
/savev2_adam_dense_48_bias_m_read_readvariableop5
1savev2_adam_dense_49_kernel_m_read_readvariableop3
/savev2_adam_dense_49_bias_m_read_readvariableop5
1savev2_adam_dense_50_kernel_m_read_readvariableop3
/savev2_adam_dense_50_bias_m_read_readvariableop5
1savev2_adam_dense_51_kernel_m_read_readvariableop3
/savev2_adam_dense_51_bias_m_read_readvariableop5
1savev2_adam_dense_52_kernel_m_read_readvariableop3
/savev2_adam_dense_52_bias_m_read_readvariableop5
1savev2_adam_dense_53_kernel_m_read_readvariableop3
/savev2_adam_dense_53_bias_m_read_readvariableop5
1savev2_adam_dense_48_kernel_v_read_readvariableop3
/savev2_adam_dense_48_bias_v_read_readvariableop5
1savev2_adam_dense_49_kernel_v_read_readvariableop3
/savev2_adam_dense_49_bias_v_read_readvariableop5
1savev2_adam_dense_50_kernel_v_read_readvariableop3
/savev2_adam_dense_50_bias_v_read_readvariableop5
1savev2_adam_dense_51_kernel_v_read_readvariableop3
/savev2_adam_dense_51_bias_v_read_readvariableop5
1savev2_adam_dense_52_kernel_v_read_readvariableop3
/savev2_adam_dense_52_bias_v_read_readvariableop5
1savev2_adam_dense_53_kernel_v_read_readvariableop3
/savev2_adam_dense_53_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_48_kernel_read_readvariableop(savev2_dense_48_bias_read_readvariableop*savev2_dense_49_kernel_read_readvariableop(savev2_dense_49_bias_read_readvariableop*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_48_kernel_m_read_readvariableop/savev2_adam_dense_48_bias_m_read_readvariableop1savev2_adam_dense_49_kernel_m_read_readvariableop/savev2_adam_dense_49_bias_m_read_readvariableop1savev2_adam_dense_50_kernel_m_read_readvariableop/savev2_adam_dense_50_bias_m_read_readvariableop1savev2_adam_dense_51_kernel_m_read_readvariableop/savev2_adam_dense_51_bias_m_read_readvariableop1savev2_adam_dense_52_kernel_m_read_readvariableop/savev2_adam_dense_52_bias_m_read_readvariableop1savev2_adam_dense_53_kernel_m_read_readvariableop/savev2_adam_dense_53_bias_m_read_readvariableop1savev2_adam_dense_48_kernel_v_read_readvariableop/savev2_adam_dense_48_bias_v_read_readvariableop1savev2_adam_dense_49_kernel_v_read_readvariableop/savev2_adam_dense_49_bias_v_read_readvariableop1savev2_adam_dense_50_kernel_v_read_readvariableop/savev2_adam_dense_50_bias_v_read_readvariableop1savev2_adam_dense_51_kernel_v_read_readvariableop/savev2_adam_dense_51_bias_v_read_readvariableop1savev2_adam_dense_52_kernel_v_read_readvariableop/savev2_adam_dense_52_bias_v_read_readvariableop1savev2_adam_dense_53_kernel_v_read_readvariableop/savev2_adam_dense_53_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :
??:?:	?@:@:@ : : @:@:	@?:?:
??:?: : : : :
??:?:	?@:@:@ : : @:@:	@?:?:
??:?:
??:?:	?@:@:@ : : @:@:	@?:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 	

_output_shapes
:@:$
 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?@: %

_output_shapes
:@:$& 

_output_shapes

:@ : '

_output_shapes
: :$( 

_output_shapes

: @: )

_output_shapes
:@:%*!

_output_shapes
:	@?:!+

_output_shapes	
:?:&,"
 
_output_shapes
:
??:!-

_output_shapes	
:?:.

_output_shapes
: 
?

?
E__inference_dense_53_layer_call_and_return_conditional_losses_1455680

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455556

inputs$
dense_48_1455540:
??
dense_48_1455542:	?#
dense_49_1455545:	?@
dense_49_1455547:@"
dense_50_1455550:@ 
dense_50_1455552: 
identity?? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_1455412?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_48_1455540dense_48_1455542*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_1455425?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_1455545dense_49_1455547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_1455442?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_1455550dense_50_1455552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_1455459x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_48_layer_call_and_return_conditional_losses_1456532

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_8_layer_call_fn_1456042
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1455986s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456072
input_1)
sequential_16_1456045:
??$
sequential_16_1456047:	?(
sequential_16_1456049:	?@#
sequential_16_1456051:@'
sequential_16_1456053:@ #
sequential_16_1456055: '
sequential_17_1456058: @#
sequential_17_1456060:@(
sequential_17_1456062:	@?$
sequential_17_1456064:	?)
sequential_17_1456066:
??$
sequential_17_1456068:	?
identity??%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_1456045sequential_16_1456047sequential_16_1456049sequential_16_1456051sequential_16_1456053sequential_16_1456055*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455466?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_1456058sequential_17_1456060sequential_17_1456062sequential_17_1456064sequential_17_1456066sequential_17_1456068*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455702?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?

b
F__inference_reshape_8_layer_call_and_return_conditional_losses_1456650

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_49_layer_call_and_return_conditional_losses_1456552

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1456372

inputs;
'dense_48_matmul_readvariableop_resource:
??7
(dense_48_biasadd_readvariableop_resource:	?:
'dense_49_matmul_readvariableop_resource:	?@6
(dense_49_biasadd_readvariableop_resource:@9
'dense_50_matmul_readvariableop_resource:@ 6
(dense_50_biasadd_readvariableop_resource: 
identity??dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/BiasAdd/ReadVariableOp?dense_49/MatMul/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?dense_50/MatMul/ReadVariableOp`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  q
flatten_8/ReshapeReshapeinputsflatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_48/MatMulMatMulflatten_8/Reshape:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_50/MatMulMatMuldense_49/Relu:activations:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? b
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*'
_output_shapes
:????????? j
IdentityIdentitydense_50/Relu:activations:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_17_layer_call_fn_1456433

inputs
unknown: @
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455792s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1456501

inputs9
'dense_51_matmul_readvariableop_resource: @6
(dense_51_biasadd_readvariableop_resource:@:
'dense_52_matmul_readvariableop_resource:	@?7
(dense_52_biasadd_readvariableop_resource:	?;
'dense_53_matmul_readvariableop_resource:
??7
(dense_53_biasadd_readvariableop_resource:	?
identity??dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0{
dense_51/MatMulMatMulinputs&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
dense_52/MatMulMatMuldense_51/Relu:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_53/MatMulMatMuldense_52/Relu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????i
dense_53/SigmoidSigmoiddense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????S
reshape_8/ShapeShapedense_53/Sigmoid:y:0*
T0*
_output_shapes
:g
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape_8/ReshapeReshapedense_53/Sigmoid:y:0 reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????m
IdentityIdentityreshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : : : : : 2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_1456512

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456280
xI
5sequential_16_dense_48_matmul_readvariableop_resource:
??E
6sequential_16_dense_48_biasadd_readvariableop_resource:	?H
5sequential_16_dense_49_matmul_readvariableop_resource:	?@D
6sequential_16_dense_49_biasadd_readvariableop_resource:@G
5sequential_16_dense_50_matmul_readvariableop_resource:@ D
6sequential_16_dense_50_biasadd_readvariableop_resource: G
5sequential_17_dense_51_matmul_readvariableop_resource: @D
6sequential_17_dense_51_biasadd_readvariableop_resource:@H
5sequential_17_dense_52_matmul_readvariableop_resource:	@?E
6sequential_17_dense_52_biasadd_readvariableop_resource:	?I
5sequential_17_dense_53_matmul_readvariableop_resource:
??E
6sequential_17_dense_53_biasadd_readvariableop_resource:	?
identity??-sequential_16/dense_48/BiasAdd/ReadVariableOp?,sequential_16/dense_48/MatMul/ReadVariableOp?-sequential_16/dense_49/BiasAdd/ReadVariableOp?,sequential_16/dense_49/MatMul/ReadVariableOp?-sequential_16/dense_50/BiasAdd/ReadVariableOp?,sequential_16/dense_50/MatMul/ReadVariableOp?-sequential_17/dense_51/BiasAdd/ReadVariableOp?,sequential_17/dense_51/MatMul/ReadVariableOp?-sequential_17/dense_52/BiasAdd/ReadVariableOp?,sequential_17/dense_52/MatMul/ReadVariableOp?-sequential_17/dense_53/BiasAdd/ReadVariableOp?,sequential_17/dense_53/MatMul/ReadVariableOpn
sequential_16/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ?
sequential_16/flatten_8/ReshapeReshapex&sequential_16/flatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
,sequential_16/dense_48/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_48_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_16/dense_48/MatMulMatMul(sequential_16/flatten_8/Reshape:output:04sequential_16/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_16/dense_48/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_16/dense_48/BiasAddBiasAdd'sequential_16/dense_48/MatMul:product:05sequential_16/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_16/dense_48/ReluRelu'sequential_16/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_16/dense_49/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_49_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_16/dense_49/MatMulMatMul)sequential_16/dense_48/Relu:activations:04sequential_16/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
-sequential_16/dense_49/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_16/dense_49/BiasAddBiasAdd'sequential_16/dense_49/MatMul:product:05sequential_16/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
sequential_16/dense_49/ReluRelu'sequential_16/dense_49/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_16/dense_50/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_50_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
sequential_16/dense_50/MatMulMatMul)sequential_16/dense_49/Relu:activations:04sequential_16/dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
-sequential_16/dense_50/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_16/dense_50/BiasAddBiasAdd'sequential_16/dense_50/MatMul:product:05sequential_16/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
sequential_16/dense_50/ReluRelu'sequential_16/dense_50/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
,sequential_17/dense_51/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_51_matmul_readvariableop_resource*
_output_shapes

: @*
dtype0?
sequential_17/dense_51/MatMulMatMul)sequential_16/dense_50/Relu:activations:04sequential_17/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
-sequential_17/dense_51/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_17/dense_51/BiasAddBiasAdd'sequential_17/dense_51/MatMul:product:05sequential_17/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
sequential_17/dense_51/ReluRelu'sequential_17/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_17/dense_52/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_52_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
sequential_17/dense_52/MatMulMatMul)sequential_17/dense_51/Relu:activations:04sequential_17/dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_17/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_17/dense_52/BiasAddBiasAdd'sequential_17/dense_52/MatMul:product:05sequential_17/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_17/dense_52/ReluRelu'sequential_17/dense_52/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_17/dense_53/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_53_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_17/dense_53/MatMulMatMul)sequential_17/dense_52/Relu:activations:04sequential_17/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_17/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_17/dense_53/BiasAddBiasAdd'sequential_17/dense_53/MatMul:product:05sequential_17/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_17/dense_53/SigmoidSigmoid'sequential_17/dense_53/BiasAdd:output:0*
T0*(
_output_shapes
:??????????o
sequential_17/reshape_8/ShapeShape"sequential_17/dense_53/Sigmoid:y:0*
T0*
_output_shapes
:u
+sequential_17/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_17/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_17/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_17/reshape_8/strided_sliceStridedSlice&sequential_17/reshape_8/Shape:output:04sequential_17/reshape_8/strided_slice/stack:output:06sequential_17/reshape_8/strided_slice/stack_1:output:06sequential_17/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_17/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_17/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
%sequential_17/reshape_8/Reshape/shapePack.sequential_17/reshape_8/strided_slice:output:00sequential_17/reshape_8/Reshape/shape/1:output:00sequential_17/reshape_8/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
sequential_17/reshape_8/ReshapeReshape"sequential_17/dense_53/Sigmoid:y:0.sequential_17/reshape_8/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????{
IdentityIdentity(sequential_17/reshape_8/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp.^sequential_16/dense_48/BiasAdd/ReadVariableOp-^sequential_16/dense_48/MatMul/ReadVariableOp.^sequential_16/dense_49/BiasAdd/ReadVariableOp-^sequential_16/dense_49/MatMul/ReadVariableOp.^sequential_16/dense_50/BiasAdd/ReadVariableOp-^sequential_16/dense_50/MatMul/ReadVariableOp.^sequential_17/dense_51/BiasAdd/ReadVariableOp-^sequential_17/dense_51/MatMul/ReadVariableOp.^sequential_17/dense_52/BiasAdd/ReadVariableOp-^sequential_17/dense_52/MatMul/ReadVariableOp.^sequential_17/dense_53/BiasAdd/ReadVariableOp-^sequential_17/dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2^
-sequential_16/dense_48/BiasAdd/ReadVariableOp-sequential_16/dense_48/BiasAdd/ReadVariableOp2\
,sequential_16/dense_48/MatMul/ReadVariableOp,sequential_16/dense_48/MatMul/ReadVariableOp2^
-sequential_16/dense_49/BiasAdd/ReadVariableOp-sequential_16/dense_49/BiasAdd/ReadVariableOp2\
,sequential_16/dense_49/MatMul/ReadVariableOp,sequential_16/dense_49/MatMul/ReadVariableOp2^
-sequential_16/dense_50/BiasAdd/ReadVariableOp-sequential_16/dense_50/BiasAdd/ReadVariableOp2\
,sequential_16/dense_50/MatMul/ReadVariableOp,sequential_16/dense_50/MatMul/ReadVariableOp2^
-sequential_17/dense_51/BiasAdd/ReadVariableOp-sequential_17/dense_51/BiasAdd/ReadVariableOp2\
,sequential_17/dense_51/MatMul/ReadVariableOp,sequential_17/dense_51/MatMul/ReadVariableOp2^
-sequential_17/dense_52/BiasAdd/ReadVariableOp-sequential_17/dense_52/BiasAdd/ReadVariableOp2\
,sequential_17/dense_52/MatMul/ReadVariableOp,sequential_17/dense_52/MatMul/ReadVariableOp2^
-sequential_17/dense_53/BiasAdd/ReadVariableOp-sequential_17/dense_53/BiasAdd/ReadVariableOp2\
,sequential_17/dense_53/MatMul/ReadVariableOp,sequential_17/dense_53/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
/__inference_sequential_16_layer_call_fn_1455588
flatten_8_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_8_input
?

?
E__inference_dense_51_layer_call_and_return_conditional_losses_1455646

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_reshape_8_layer_call_fn_1456637

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_1455699d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_signature_wrapper_1456311
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: @
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_1455399s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
#__inference__traced_restore_1456953
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: 6
"assignvariableop_5_dense_48_kernel:
??/
 assignvariableop_6_dense_48_bias:	?5
"assignvariableop_7_dense_49_kernel:	?@.
 assignvariableop_8_dense_49_bias:@4
"assignvariableop_9_dense_50_kernel:@ /
!assignvariableop_10_dense_50_bias: 5
#assignvariableop_11_dense_51_kernel: @/
!assignvariableop_12_dense_51_bias:@6
#assignvariableop_13_dense_52_kernel:	@?0
!assignvariableop_14_dense_52_bias:	?7
#assignvariableop_15_dense_53_kernel:
??0
!assignvariableop_16_dense_53_bias:	?#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: >
*assignvariableop_21_adam_dense_48_kernel_m:
??7
(assignvariableop_22_adam_dense_48_bias_m:	?=
*assignvariableop_23_adam_dense_49_kernel_m:	?@6
(assignvariableop_24_adam_dense_49_bias_m:@<
*assignvariableop_25_adam_dense_50_kernel_m:@ 6
(assignvariableop_26_adam_dense_50_bias_m: <
*assignvariableop_27_adam_dense_51_kernel_m: @6
(assignvariableop_28_adam_dense_51_bias_m:@=
*assignvariableop_29_adam_dense_52_kernel_m:	@?7
(assignvariableop_30_adam_dense_52_bias_m:	?>
*assignvariableop_31_adam_dense_53_kernel_m:
??7
(assignvariableop_32_adam_dense_53_bias_m:	?>
*assignvariableop_33_adam_dense_48_kernel_v:
??7
(assignvariableop_34_adam_dense_48_bias_v:	?=
*assignvariableop_35_adam_dense_49_kernel_v:	?@6
(assignvariableop_36_adam_dense_49_bias_v:@<
*assignvariableop_37_adam_dense_50_kernel_v:@ 6
(assignvariableop_38_adam_dense_50_bias_v: <
*assignvariableop_39_adam_dense_51_kernel_v: @6
(assignvariableop_40_adam_dense_51_bias_v:@=
*assignvariableop_41_adam_dense_52_kernel_v:	@?7
(assignvariableop_42_adam_dense_52_bias_v:	?>
*assignvariableop_43_adam_dense_53_kernel_v:
??7
(assignvariableop_44_adam_dense_53_bias_v:	?
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_48_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_48_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_49_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_49_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_50_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_50_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_51_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_51_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_52_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_52_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_53_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_53_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_48_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_48_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_49_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_49_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_50_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_50_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_51_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_51_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_52_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_52_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_53_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_53_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_48_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_48_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_49_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_49_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_50_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_50_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_51_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_51_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_52_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_52_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_53_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_53_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_model
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
 iter

!beta_1

"beta_2
	#decay
$learning_rate%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?"
	optimizer
v
%0
&1
'2
(3
)4
*5
+6
,7
-8
.9
/10
011"
trackable_list_wrapper
v
%0
&1
'2
(3
)4
*5
+6
,7
-8
.9
/10
011"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_autoencoder_8_layer_call_fn_1455925
/__inference_autoencoder_8_layer_call_fn_1456137
/__inference_autoencoder_8_layer_call_fn_1456166
/__inference_autoencoder_8_layer_call_fn_1456042?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456223
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456280
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456072
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456102?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference__wrapped_model_1455399input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
6serving_default"
signature_map
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
?

%kernel
&bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_16_layer_call_fn_1455481
/__inference_sequential_16_layer_call_fn_1456328
/__inference_sequential_16_layer_call_fn_1456345
/__inference_sequential_16_layer_call_fn_1455588?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1456372
J__inference_sequential_16_layer_call_and_return_conditional_losses_1456399
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455608
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455628?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?

+kernel
,bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

-kernel
.bias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_17_layer_call_fn_1455717
/__inference_sequential_17_layer_call_fn_1456416
/__inference_sequential_17_layer_call_fn_1456433
/__inference_sequential_17_layer_call_fn_1455824?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1456467
J__inference_sequential_17_layer_call_and_return_conditional_losses_1456501
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455844
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455864?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
#:!
??2dense_48/kernel
:?2dense_48/bias
": 	?@2dense_49/kernel
:@2dense_49/bias
!:@ 2dense_50/kernel
: 2dense_50/bias
!: @2dense_51/kernel
:@2dense_51/bias
": 	@?2dense_52/kernel
:?2dense_52/bias
#:!
??2dense_53/kernel
:?2dense_53/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_1456311input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_flatten_8_layer_call_fn_1456506?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_8_layer_call_and_return_conditional_losses_1456512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_48_layer_call_fn_1456521?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_48_layer_call_and_return_conditional_losses_1456532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_49_layer_call_fn_1456541?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_49_layer_call_and_return_conditional_losses_1456552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_50_layer_call_fn_1456561?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_50_layer_call_and_return_conditional_losses_1456572?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_51_layer_call_fn_1456581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_51_layer_call_and_return_conditional_losses_1456592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_52_layer_call_fn_1456601?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_52_layer_call_and_return_conditional_losses_1456612?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_53_layer_call_fn_1456621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_53_layer_call_and_return_conditional_losses_1456632?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_reshape_8_layer_call_fn_1456637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_8_layer_call_and_return_conditional_losses_1456650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_48/kernel/m
!:?2Adam/dense_48/bias/m
':%	?@2Adam/dense_49/kernel/m
 :@2Adam/dense_49/bias/m
&:$@ 2Adam/dense_50/kernel/m
 : 2Adam/dense_50/bias/m
&:$ @2Adam/dense_51/kernel/m
 :@2Adam/dense_51/bias/m
':%	@?2Adam/dense_52/kernel/m
!:?2Adam/dense_52/bias/m
(:&
??2Adam/dense_53/kernel/m
!:?2Adam/dense_53/bias/m
(:&
??2Adam/dense_48/kernel/v
!:?2Adam/dense_48/bias/v
':%	?@2Adam/dense_49/kernel/v
 :@2Adam/dense_49/bias/v
&:$@ 2Adam/dense_50/kernel/v
 : 2Adam/dense_50/bias/v
&:$ @2Adam/dense_51/kernel/v
 :@2Adam/dense_51/bias/v
':%	@?2Adam/dense_52/kernel/v
!:?2Adam/dense_52/bias/v
(:&
??2Adam/dense_53/kernel/v
!:?2Adam/dense_53/bias/v?
"__inference__wrapped_model_1455399}%&'()*+,-./04?1
*?'
%?"
input_1?????????
? "7?4
2
output_1&?#
output_1??????????
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456072s%&'()*+,-./08?5
.?+
%?"
input_1?????????
p 
? ")?&
?
0?????????
? ?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456102s%&'()*+,-./08?5
.?+
%?"
input_1?????????
p
? ")?&
?
0?????????
? ?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456223m%&'()*+,-./02?/
(?%
?
x?????????
p 
? ")?&
?
0?????????
? ?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_1456280m%&'()*+,-./02?/
(?%
?
x?????????
p
? ")?&
?
0?????????
? ?
/__inference_autoencoder_8_layer_call_fn_1455925f%&'()*+,-./08?5
.?+
%?"
input_1?????????
p 
? "???????????
/__inference_autoencoder_8_layer_call_fn_1456042f%&'()*+,-./08?5
.?+
%?"
input_1?????????
p
? "???????????
/__inference_autoencoder_8_layer_call_fn_1456137`%&'()*+,-./02?/
(?%
?
x?????????
p 
? "???????????
/__inference_autoencoder_8_layer_call_fn_1456166`%&'()*+,-./02?/
(?%
?
x?????????
p
? "???????????
E__inference_dense_48_layer_call_and_return_conditional_losses_1456532^%&0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_48_layer_call_fn_1456521Q%&0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_49_layer_call_and_return_conditional_losses_1456552]'(0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ~
*__inference_dense_49_layer_call_fn_1456541P'(0?-
&?#
!?
inputs??????????
? "??????????@?
E__inference_dense_50_layer_call_and_return_conditional_losses_1456572\)*/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? }
*__inference_dense_50_layer_call_fn_1456561O)*/?,
%?"
 ?
inputs?????????@
? "?????????? ?
E__inference_dense_51_layer_call_and_return_conditional_losses_1456592\+,/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????@
? }
*__inference_dense_51_layer_call_fn_1456581O+,/?,
%?"
 ?
inputs????????? 
? "??????????@?
E__inference_dense_52_layer_call_and_return_conditional_losses_1456612]-./?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_dense_52_layer_call_fn_1456601P-./?,
%?"
 ?
inputs?????????@
? "????????????
E__inference_dense_53_layer_call_and_return_conditional_losses_1456632^/00?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_53_layer_call_fn_1456621Q/00?-
&?#
!?
inputs??????????
? "????????????
F__inference_flatten_8_layer_call_and_return_conditional_losses_1456512]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? 
+__inference_flatten_8_layer_call_fn_1456506P3?0
)?&
$?!
inputs?????????
? "????????????
F__inference_reshape_8_layer_call_and_return_conditional_losses_1456650]0?-
&?#
!?
inputs??????????
? ")?&
?
0?????????
? 
+__inference_reshape_8_layer_call_fn_1456637P0?-
&?#
!?
inputs??????????
? "???????????
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455608u%&'()*D?A
:?7
-?*
flatten_8_input?????????
p 

 
? "%?"
?
0????????? 
? ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1455628u%&'()*D?A
:?7
-?*
flatten_8_input?????????
p

 
? "%?"
?
0????????? 
? ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1456372l%&'()*;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1456399l%&'()*;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
/__inference_sequential_16_layer_call_fn_1455481h%&'()*D?A
:?7
-?*
flatten_8_input?????????
p 

 
? "?????????? ?
/__inference_sequential_16_layer_call_fn_1455588h%&'()*D?A
:?7
-?*
flatten_8_input?????????
p

 
? "?????????? ?
/__inference_sequential_16_layer_call_fn_1456328_%&'()*;?8
1?.
$?!
inputs?????????
p 

 
? "?????????? ?
/__inference_sequential_16_layer_call_fn_1456345_%&'()*;?8
1?.
$?!
inputs?????????
p

 
? "?????????? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455844t+,-./0??<
5?2
(?%
dense_51_input????????? 
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1455864t+,-./0??<
5?2
(?%
dense_51_input????????? 
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1456467l+,-./07?4
-?*
 ?
inputs????????? 
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_1456501l+,-./07?4
-?*
 ?
inputs????????? 
p

 
? ")?&
?
0?????????
? ?
/__inference_sequential_17_layer_call_fn_1455717g+,-./0??<
5?2
(?%
dense_51_input????????? 
p 

 
? "???????????
/__inference_sequential_17_layer_call_fn_1455824g+,-./0??<
5?2
(?%
dense_51_input????????? 
p

 
? "???????????
/__inference_sequential_17_layer_call_fn_1456416_+,-./07?4
-?*
 ?
inputs????????? 
p 

 
? "???????????
/__inference_sequential_17_layer_call_fn_1456433_+,-./07?4
-?*
 ?
inputs????????? 
p

 
? "???????????
%__inference_signature_wrapper_1456311?%&'()*+,-./0??<
? 
5?2
0
input_1%?"
input_1?????????"7?4
2
output_1&?#
output_1?????????