import React from 'react';

import { Grid, makeStyles, Typography, TextField, Button } from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
    root: {
        display: "flex",
        flexDirection: "row-reverse"
    },
    btn: {
        background: " linear-gradient(#3585da 0%, #47a4e1 48.19%, #59c1e8 100%)",
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
        borderRadius: 10,
        color: "#fff",
        fontWeight: "bold",
    },
    txt: {
        color: "#1061b0",
        fontWeight: "bold",
    }
}))

export default function Startsession() {
    const classes = useStyles();
    return (
        <Grid item>
            <Typography variant="h6" className={classes.txt}>Hi! ABDUL HADI </Typography>
            <Typography variant="h6" className={classes.txt}>WELCOME TO VIRTUAL CLINIC </Typography>
            <Grid item className={classes.root}>
                <Button className={classes.btn}>START SESSION</Button>
            </Grid>
        </Grid>
    )
}