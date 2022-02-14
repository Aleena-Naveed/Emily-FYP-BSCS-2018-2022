import React from "react";

import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";

import Drawer from "../components/drawer";
import Setting from "../components/Settings";

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
        background: "#f5f5f5",
        minHeight: "100vh",
    },
    settingsGrid: {
        margin: "auto",
    },
}))

export default function Settings() {

    const classes = useStyles();

    return (
        <Grid container className={classes.root}>
            <Grid item xs={1}>
                <Drawer />
            </Grid>
            <Grid item xs={10} className={classes.settingsGrid}>
                <Setting />
            </Grid>
        </Grid>
    )
}