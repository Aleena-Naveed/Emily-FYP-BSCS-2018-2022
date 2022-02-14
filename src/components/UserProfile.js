import React from 'react';

import { Grid, makeStyles, Typography, TextField } from '@material-ui/core';
import { useSelector } from 'react-redux';

const useStyles = makeStyles((theme) => ({
    root: {
        borderRadius: 30,
        background: "#fff",
        boxShadow: " 0px 3px 6px rgba(0, 0, 0, 0.16)",
        textAlign: "center",
        padding: 30,
        height: "85vh"
    },
    heading: {
        color: "#1061b0",
        textDecoration: "underline",
        fontWeight: "bold",
        margin: "auto",
    },
    textfiledcontainer: {
        justifyContent: "center",
    },
    textfieldstyle: {
        borderRadius: 10,
        background: "#fff",
        boxShadow: "6px 6px 10px rgba(0, 0, 0, 0.16)",
        width: "95%",
    },
    inputProps: {
        color: "#1061b0",
        fontWeight: "bold",
    }
}))

export default function Userprofile() {
    const classes = useStyles();
    const user = useSelector((state) => state.states.user)
    console.log(user)
    return (
        <Grid container className={classes.root}>
            <Typography variant="h5" className={classes.heading}>User Profile</Typography>
            <Grid container className={classes.textfiledcontainer}>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="PATIENT ID"
                        defaultValue={user._id}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="USERNAME"
                        defaultValue={user.username}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="FULL NAME"
                        defaultValue={user.name.toUpperCase()}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
            </Grid>
            <Grid container className={classes.textfiledcontainer}>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="EMAIL ADDRESS"
                        defaultValue={user.email}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="CONTACT NUMBER"
                        defaultValue={user.contact}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="OCCUPATION"
                        defaultValue={user.occupation}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
            </Grid>
            <Grid container className={classes.textfiledcontainer}>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="COUNTRY"
                        defaultValue={user.country.toUpperCase()}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="GENDER"
                        defaultValue={user.gender}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <TextField
                        fullWidth
                        id="outlined-read-only-input"
                        label="DATE OF BIRTH"
                        defaultValue={user.dob}
                        InputProps={{
                            readOnly: true,
                        }}
                        InputLabelProps={{
                            className: classes.inputProps
                        }}
                        inputProps={{
                            className: classes.inputProps,
                        }}
                        variant="outlined"
                        className={classes.textfieldstyle}
                    />
                </Grid>
            </Grid>

        </Grid>
    )
}