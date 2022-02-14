import React from "react";
import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";
import { Typography } from "@material-ui/core";
import IconButton from '@material-ui/core/IconButton';

import VisibilityOutlinedIcon from '@material-ui/icons/VisibilityOutlined';
import CloudDownloadOutlinedIcon from '@material-ui/icons/CloudDownloadOutlined';

const useStyles = makeStyles({
  root: {
    margin: "auto",
  },
  DialogBox: {
    width: "100%",
    borderRadius: "30px",
    background: "#fff",
    boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
  },
  sameinfont: {
    fontWeight: "bold",
    textDecoration: "underline",
    color: "#3585da",
    textShadow: "0px 3px 6px rgba(0, 0, 0, 0.16)",
  },
  TDialogbox: {
    width: "100%",
    background: "#fff",
    boxShadow: "6px 6px 10px rgba(0, 0, 0, 0.16)",
    borderRadius: "12px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    marginTop: "20px",
  },
  Tablecontentbox: {
    width: "100%",
    borderRadius: "10px",
    background: "linear-gradient(45deg,#59c1e8 0%, #59c1e8 100%)",
    filter: "drop-shadow(6px 6px 10px rgba(0, 0, 0, 0.16))",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: "35px",
    padding: "0px 10px 0px 10px",
  },
  Reportheader: {
    width: "100%",
    borderRadius: "12px",
    background: "linear-gradient(45deg,#59c1e8 0%, #3585da 100%)",
    boxShadow: "6px 6px 10px rgba(0, 0, 0, 0.16)",
    display: "flex",
    alignItems: "center",
    marginTop: "35px",
    padding: "12px",
  },
  TableContentFont: {
    fontWeight: "bold",
    fontSize: "14px",
    color: "#fff",
  },
  reportlistcontainer: {
    marginBottom: "30px",
    justifyContent: "center",
  }
});

function Report() {
  const classes = useStyles();
  return (
    <Grid item xs={11} className={classes.Tablecontentbox}>
      <Grid container style={{ alignItems: "center" }}>
        <Grid item xs={4} sm={2} style={{ textAlign: 'start' }}>
          <Typography className={classes.TableContentFont}>01</Typography>
        </Grid>
        <Grid item xs={8} sm={4} style={{ textAlign: 'start' }}>
          <Typography className={classes.TableContentFont}>
            EMILY_3581_240821
          </Typography>
        </Grid>
        <Grid item xs={4} sm={2} style={{ textAlign: 'start' }}>
          <Typography className={classes.TableContentFont}>24/08/2021</Typography>
        </Grid>
        <Grid item xs={4} sm={2} style={{ textAlign: 'start' }}>
          <Typography className={classes.TableContentFont}>10:27:31 AM</Typography>
        </Grid>
        <Grid item xs={12} sm={2} style={{ textAlign: 'start' }}>
          <IconButton>
            <CloudDownloadOutlinedIcon style={{ color: "#fff" }} />
          </IconButton>
          <Typography display="inline" style={{ color: "#fff", fontWeight: "bold" }}>|</Typography>
          <IconButton>
            <VisibilityOutlinedIcon style={{ color: "#fff" }} />
          </IconButton>
        </Grid>
      </Grid>
    </Grid>
  );
}
export default function Reports() {
  const classes = useStyles();
  return (
    <Grid container className={classes.root}>
      <Grid item xs={12} className={classes.DialogBox}>
        <Typography variant="h4" className={classes.sameinfont}>
          REPORTS
        </Typography>

        <Grid item xs={12} sm={11} className={classes.TDialogbox}>
          <Grid item xs={11} className={classes.Reportheader}>
            <Grid container>
              <Grid item xs={4} sm={2} style={{ textAlign: 'start' }}>
                <Typography className={classes.TableContentFont}>SR</Typography>
              </Grid>
              <Grid item xs={8} sm={4} style={{ textAlign: 'start' }}>
                <Typography className={classes.TableContentFont}>
                  REPORT NAME
                </Typography>
              </Grid>

              <Grid item xs={4} sm={2} style={{ textAlign: 'start' }}>
                <Typography className={classes.TableContentFont}>DATE</Typography>
              </Grid>
              <Grid item xs={4} sm={2} style={{ textAlign: 'start' }}>
                <Typography className={classes.TableContentFont}>TIME</Typography>
              </Grid>
              <Grid item xs={12} sm={2} style={{ textAlign: 'start' }}>
                <Typography className={classes.TableContentFont}>
                  ACTIONS
                </Typography>
              </Grid>
            </Grid>
          </Grid>
          <Grid container className={classes.reportlistcontainer}>
            <Report />
            <Report />
            <Report />
            <Report />
            <Report />
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}
