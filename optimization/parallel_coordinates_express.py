import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
ynames = [
	"time_steps",
	"state_depth",
	"filters_downsampling",
	"downsampling_layers",
	"conv_layers",
	"hidden_layers",
	"filters_upsampling",
	"kernel_regularizer_amp",
	"bias_regularizer_amp",
	"batch_norm",
	"activation",
	"initial_learning_rate",
	"beta_1",  # Adam update in RIM
	"beta_2",
    "train_loss",
    "chi_squared_train"
    ]

df = pd.read_csv("../results/scores_1.csv", names=ynames)
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['train_loss'],
                   colorscale = 'Jet',
                   showscale = True,
                   cmin = 0,
                   cmax = 0.6),
        dimensions = list([
            dict(range = [6, 12],
                 # constraintrange = [100000,150000],
                 label = "Time steps", 
                 values = df['time_steps']),
            dict(label= "State Depth",
                values=df["state_depth"]),
            dict(label = "Loss", 
                constraintrange = [0, 0.5],
                values = df["train_loss"]),
            # dict(range = [0,700000],
                 # label = 'Block Width', values = df['blockWidth']),
            # dict(tickvals = [0,0.5,1,2,3],
                 # ticktext = ['A','AB','B','Y','Z'],
                 # label = 'Cyclinder Material', values = df['cycMaterial']),
            # dict(range = [-1,4],
                 # tickvals = [0,1,2,3],
                 # label = 'Block Material', values = df['blockMaterial']),
            # dict(range = [134,3154],
                 # visible = True,
                 # label = 'Total Weight', values = df['totalWeight']),
            # dict(range = [9,19984],
                 # label = 'Assembly Penalty Wt', values = df['assemblyPW']),
            # dict(range = [49000,568000],
                 # label = 'Height st Width', values = df['HstW'])
            ])
    )
)
fig.show()


# fig = px.parallel_coordinates(df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]], color="train_loss", 
        # # labels={"species_id": "Species",
                # # "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
                # # "petal_width": "Petal Width", "petal_length": "Petal Length", },
                             # color_continuous_scale=px.colors.diverging.Tealrose)
# fig.show()

