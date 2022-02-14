import React from 'react';

import { Button, Grid, makeStyles, Typography, TextField } from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
    root: {
        padding: 20,
        alignSelf: "center",
        background: "#fff",
        borderRadius: 30,
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
    },
    heading: {
        color: "#1061b0",
        textDecoration: "underline",
        fontWeight: "bold",
        textAlign: "center",
    },
    subheading: {
        color: "#3585da",
        textDecoration: "underline",
        fontWeight: "bold",
    },
    row: {
        padding: "10px 50px 10px 50px",
        marginTop: 20,
    },
    textfiledcontainer: {
        padding: "20px 10px 30px 10px",
        marginTop: 25,
        borderRadius: 10,
        background: "#fff",
        boxShadow: "6px 6px 10px rgba(0, 0, 0, 0.16)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
    },
    textfieldstyle: {
        borderRadius: 10,
        background: "#fff",
        width: "95%",
    },
    inputProps: {
        color: "#3585da",
        fontWeight: "bold",
    },
    btn: {
        borderRadius: 20,
        background: "#3585da",
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
        color: "#fff",
        fontWeight: "bold",
        width: "90%",
        "&:hover": {
            background: "rgba(53,133,218,0.7)",
        }
    }
}))

export default function Settings() {
    const classes = useStyles();
    const [verify, setverify] = React.useState(false)
    return (
        <Grid className={classes.root}>
            <Typography variant="h5" className={classes.heading}>SETTINGS</Typography>
            <Grid item className={classes.row}>
                <Typography variant="body1" className={classes.subheading}>CHANGE CONTACT NUMBER</Typography>
                <Grid container className={classes.textfiledcontainer}>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            id="standard-number"
                            label="CONTACT NUMBER"
                            type="text"
                            InputLabelProps={{
                                shrink: true,
                                className: classes.inputProps
                            }}
                            inputProps={{
                                className: classes.inputProps,
                            }}
                            className={classes.textfieldstyle}
                        />
                    </Grid>
                    {
                        verify ?
                            <Grid item xs={12} sm={3}>
                                <TextField
                                    fullWidth
                                    id="standard-number"
                                    label="OTP (RESEND IN 30 SECONDS)"
                                    type="text"
                                    InputLabelProps={{
                                        shrink: true,
                                        className: classes.inputProps
                                    }}
                                    inputProps={{
                                        className: classes.inputProps,
                                    }}
                                    className={classes.textfieldstyle}
                                />
                            </Grid>
                            :
                            <Grid item sm={3} />
                    }
                    <Grid item sm={3} />
                    <Grid xs={12} sm={3} style={{ alignItems: "center" }}>
                        <Button
                            className={classes.btn}
                            fullWidth
                            onClick={() => setverify(!verify)}
                        >{verify ? "UPDATE" : "SEND OTP"}</Button>
                    </Grid>
                </Grid>
            </Grid>
            <Grid item className={classes.row}>
                <Typography variant="body1" className={classes.subheading}>CHANGE EMAIL</Typography>
                <Grid container className={classes.textfiledcontainer}>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            id="standard-number"
                            label="Email Address"
                            type="text"
                            InputLabelProps={{
                                shrink: true,
                                className: classes.inputProps
                            }}
                            inputProps={{
                                className: classes.inputProps,
                            }}
                            className={classes.textfieldstyle}
                        />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            id="standard-number"
                            label="Confirm Email Address"
                            type="text"
                            InputLabelProps={{
                                shrink: true,
                                className: classes.inputProps
                            }}
                            inputProps={{
                                className: classes.inputProps,
                            }}
                            className={classes.textfieldstyle}
                        />
                    </Grid>
                    <Grid item sm={3} />
                    <Grid xs={12} sm={3} style={{ alignItems: "center" }}>
                        <Button className={classes.btn} fullWidth>UPDATE</Button>
                    </Grid>
                </Grid>
            </Grid>
            <Grid item className={classes.row}>
                <Typography variant="body1" className={classes.subheading}>CHANGE PASSWORD</Typography>
                <Grid container className={classes.textfiledcontainer}>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            id="standard-number"
                            label="Previous Password"
                            type="text"
                            InputLabelProps={{
                                shrink: true,
                                className: classes.inputProps
                            }}
                            inputProps={{
                                className: classes.inputProps,
                            }}
                            className={classes.textfieldstyle}
                        />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            id="standard-number"
                            label="New Password"
                            type="text"
                            InputLabelProps={{
                                shrink: true,
                                className: classes.inputProps
                            }}
                            inputProps={{
                                className: classes.inputProps,
                            }}
                            className={classes.textfieldstyle}
                        />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            id="standard-number"
                            label="Confirm Password"
                            type="text"
                            InputLabelProps={{
                                shrink: true,
                                className: classes.inputProps
                            }}
                            inputProps={{
                                className: classes.inputProps,
                            }}
                            className={classes.textfieldstyle}
                        />
                    </Grid>
                    <Grid xs={12} sm={3} style={{ alignItems: "center" }}>
                        <Button className={classes.btn} fullWidth>UPDATE</Button>
                    </Grid>
                </Grid>
            </Grid>
        </Grid>
    )
}