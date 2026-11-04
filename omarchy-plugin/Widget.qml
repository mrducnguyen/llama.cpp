// Status for a llama.cpp router server (one started with --models-preset):
// which models it keeps loaded, the GPU card each sits on, and the port
// answering for it. Start / stop / restart go through the systemd user unit
// that runs start_server.sh.
//
// Model state comes from GET /models and GET /props, issued through curl with
// an argv array, so a base URL coming from shell.json cannot be reinterpreted
// as a shell command.
//
// The speed line under each model is read from the unit's journal. The router
// prefixes every child line with the child's port, and port is what ties a
// timing line back to a model. journalctl does the first filter (--grep) so the
// tail handed to this widget is timing lines only, otherwise a busy model's
// generation lines push the idle one out of the window.

import QtQuick
import QtQuick.Controls
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Commons as Commons
import qs.Ui

Panel {
  id: root

  moduleName: "io.github.ducn.llama-router"
  ipcTarget: "io.github.ducn.llama-router"

  // the bar sizes a widget from its implicit extent, not from its children
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  readonly property string baseUrl: String(setting("baseUrl", "http://127.0.0.1:14100")).replace(/\/+$/, "")
  readonly property int pollSec: Math.min(120, Math.max(3, parseInt(setting("pollIntervalSec", 10), 10) || 10))
  readonly property string barMode: String(setting("barMode", "count"))
  readonly property string barGlyph: String(setting("barIcon", "\ue907"))

  // the brand marks live in the "omarchy" icon font at U+E900..U+E90F (U+E907 is
  // the ollama llama); the bar's Nerd Font maps those slots to other icons
  readonly property bool barGlyphIsBrand: {
    var c = root.barGlyph.charCodeAt(0)
    return c >= 0xE900 && c <= 0xE90F
  }
  readonly property string barGlyphFont: root.barGlyphIsBrand ? "omarchy" : root.fontFamily
  readonly property bool showIdle: String(setting("showIdle", false)) === "true"
  readonly property string serviceUnit: String(setting("serviceUnit", "llama-router.service"))

  // how many matching journal lines to read per poll; the filter drops load
  // noise, so this is mostly timing lines
  readonly property int logLines: 2000

  // timing lines of a child, plus the router lines that name which port a model
  // answers on, plus error lines
  readonly property string logFilter: "eval time|n_gen =|prompt processing|draft acceptance"
    + "|stop processing|proxying request to model|spawning server instance with name|[0-9] E "

  // One row per model the router reports
  property var models: []
  property int maxInstances: 0
  property bool isRouter: false
  property string buildInfo: ""
  property bool reachable: false
  property string updatedAt: ""

  // port -> timing summary read from the journal, plus the model name -> port
  // pairs the router logs itself
  property var logStats: ({})
  property var logPorts: ({})

  // systemd's own words: active, inactive, failed, activating
  property string unitState: "unknown"
  property bool actionRunning: false
  property string actionNote: ""
  property bool confirmStop: false
  readonly property bool unitUp: root.unitState === "active" || root.unitState === "activating"

  readonly property color foreground: barForeground
  readonly property color dim: Qt.darker(foreground, 1.7)
  readonly property string fontFamily: bar ? bar.fontFamily : Style.font.family
  // up = theme text, coming up = accent, failed = urgent, stopped = dimmed
  readonly property color statusColor: root.reachable ? root.foreground
    : (root.unitState === "active" || root.unitState === "activating") ? Commons.Color.accent
    : root.unitState === "failed" ? Commons.Color.urgent : root.dim

  readonly property var loaded: root.models.filter(function (m) {
    return m.status === "loaded"
  })
  readonly property var busy: root.models.filter(function (m) {
    return m.status === "loading" || m.status === "sleeping"
  })
  readonly property var idle: root.models.filter(function (m) {
    return m.status === "unloaded" || m.status === "downloaded"
  })
  // Anything not idle is either up, coming up, or asleep: it holds a slot
  readonly property var active: root.models.filter(function (m) {
    return m.status !== "unloaded" && m.status !== "downloaded"
  })
  readonly property var cardGroups: groups()
  readonly property string cardTags: {
    var tags = []
    for (var i = 0; i < cardGroups.length; i++)
      tags.push(cardTag(cardGroups[i].device))
    return tags.join("  ")
  }
  readonly property string statusText: {
    if (root.reachable) {
      var text = root.loaded.length + " loaded"
      if (root.busy.length > 0)
        text = text + ", " + root.busy.length + " busy"
      if (root.maxInstances > 0)
        text = text + "  (limit " + root.maxInstances + ")"
      if (!root.isRouter)
        text = text + "  [not a router]"
      return text
    }
    if (root.unitState === "active")
      return "Unit is up, but " + root.baseUrl + " is not answering yet"
    if (root.unitState === "activating")
      return "Starting: " + root.baseUrl
    return root.baseUrl + " is not running (" + root.unitState + ")"
  }

  // CUDA0 -> C0, ROCm0 -> R0, Vulkan2 -> V2, unpinned -> any
  function cardTag(device) {
    if (device === "")
      return "any"
    if (device.indexOf("CPU") === 0)
      return "cpu"
    var digits = device.replace(/[^0-9]/g, "")
    return digits === "" ? device : device.charAt(0).toUpperCase() + digits
  }

  function shortName(id) {
    var i = id.lastIndexOf("/")
    return i >= 0 ? id.substring(i + 1) : id
  }

  function trim(text, width) {
    return text.length > width ? text.substring(0, width - 3) + "..." : text
  }

  // The child argv is the only place the router exposes a model's card and
  // port, and an idle entry carries the command it would start with.
  function flagValue(args, flag) {
    for (var i = 0; i + 1 < args.length; i++) {
      if (args[i] === flag)
        return String(args[i + 1])
    }
    return ""
  }

  function parseJson(text) {
    try {
      return JSON.parse(String(text).trim())
    } catch (e) {
      return null
    }
  }

  function applyModels(exitCode, text) {
    var body = exitCode === 0 ? parseJson(text) : null
    if (!body || !Array.isArray(body.data)) {
      root.reachable = false
      return
    }
    var rows = []
    for (var i = 0; i < body.data.length; i++) {
      var m = body.data[i]
      var status = m.status && m.status.value ? String(m.status.value) : "unknown"
      var args = m.status && Array.isArray(m.status.args) ? m.status.args : []
      var port = flagValue(args, "--port")
      rows.push({
        "name": shortName(String(m.id)),
        "id": String(m.id),
        "status": status,
        "port": port,
        "device": flagValue(args, "--device"),
        // the router spawns children with port 0 and lets them pick one
        "addr": port === "0" ? "" : flagValue(args, "--host") + ":" + port,
        "ctx": flagValue(args, "--ctx-size")
      })
    }
    rows.sort(function (a, b) {
      return a.device === b.device ? a.name.localeCompare(b.name) : a.device.localeCompare(b.device)
    })
    root.models = rows
    root.reachable = true
  }

  // journal -o short-unix renders a line as
  //   <epoch.us> host comm[pid]: [port] <hh.mm.sss.mmm> L text
  // where <port> is the child instance the line came from and L the level.
  function applyLogs(exitCode, text) {
    var stats = {}
    var ports = {}
    if (exitCode === 0) {
      var lines = String(text).split("\n")
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i]
        var stamp = /^(\d+\.\d+)\s/.exec(line)
        if (!stamp)
          continue
        var t = parseFloat(stamp[1])
        var m
        if ((m = /spawning server instance with name=(\S+) on port (\d+)/.exec(line))
          || (m = /proxying (?:request|stream resume) to model (.+) on port (\d+)/.exec(line))) {
          ports[m[1]] = m[2]
          continue
        }
        var pm = /:\s+\[(\d+)\]\s/.exec(line)
        if (!pm)
          continue
        var port = pm[1]
        if (stats[port] === undefined)
          stats[port] = { "pp": 0, "tg": 0, "acc": -1, "ts": 0, "err": "", "errTs": 0 }
        var s = stats[port]
        if ((m = /prompt processing, n_tokens =\s*\d+, progress = [\d.]+, .*\/ ([\d.]+) tokens per second/.exec(line))) {
          s.pp = parseFloat(m[1])
          s.ts = t
        } else if ((m = /n_gen =\s*\d+, tg = [\d.]+ t\/s, tg_3s = ([\d.]+) t\/s/.exec(line))) {
          s.tg = parseFloat(m[1])
          s.ts = t
        } else if ((m = /prompt eval time =.*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)/.exec(line))) {
          s.pp = parseFloat(m[1])
          s.ts = t
        } else if ((m = /eval time =.*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)/.exec(line))) {
          s.tg = parseFloat(m[1])
          s.ts = t
        } else if ((m = /draft acceptance = ([\d.]+)/.exec(line))) {
          s.acc = parseFloat(m[1])
        } else if (line.indexOf("stop processing:") >= 0) {
          s.ts = t
        }
        // the level letter follows the child's uptime stamp, e.g. "22.55.751.656 E "
        var lv = /\d{1,2}\.\d{2}\.\d{3}\.\d{3} ([EW]) /.exec(line)
        if (lv) {
          s.err = line.slice(lv.index + lv[0].length).trim().substring(0, 140)
          s.errTs = t
        }
      }
    }
    root.logStats = stats
    root.logPorts = ports
  }

  function applyProps(exitCode, text) {
    var body = exitCode === 0 ? parseJson(text) : null
    if (!body)
      return
    root.isRouter = body.role === "router"
    root.maxInstances = body.max_instances > 0 ? body.max_instances : 0
    root.buildInfo = body.build_info ? String(body.build_info) : ""
  }

  function groups() {
    var out = []
    for (var i = 0; i < root.active.length; i++) {
      var m = root.active[i]
      var found = null
      for (var j = 0; j < out.length; j++) {
        if (out[j].device === m.device)
          found = out[j]
      }
      if (!found) {
        found = { "device": m.device, "rows": [] }
        out.push(found)
      }
      found.rows.push(m)
    }
    out.sort(function (a, b) {
      return a.device.localeCompare(b.device)
    })
    return out
  }

  // timing summary for a model row: the port from its argv, else the port the
  // router used the last time it proxied to that name
  function statsFor(row) {
    var port = row.port
    if (root.logStats[port] === undefined && root.logPorts[row.name] !== undefined)
      port = root.logPorts[row.name]
    return root.logStats[port]
  }

  function ageText(sec) {
    if (sec < 5)
      return "now"
    if (sec < 120)
      return Math.round(sec) + "s ago"
    if (sec < 7200)
      return Math.round(sec / 60) + "m ago"
    return Math.round(sec / 3600) + "h ago"
  }

  // one line under a model: prompt and generation speed, from its last request
  function speedLine(row) {
    var s = root.statsFor(row)
    if (!s)
      return row.status === "loaded" ? "no request logged yet" : ""
    if (s.err !== "" && s.errTs > s.ts)
      return s.err
    var parts = []
    if (s.pp > 0)
      parts.push("pp " + s.pp.toFixed(1) + " t/s")
    if (s.tg > 0)
      parts.push("tg " + s.tg.toFixed(1) + " t/s")
    if (s.acc >= 0)
      parts.push("draft " + Math.round(s.acc * 100) + "%")
    if (parts.length === 0)
      return ""
    parts.push(root.ageText(Date.now() / 1000 - s.ts))
    return parts.join("   ")
  }

  // a model still working prints a timing line every few seconds, so recent
  // lines are what makes a readout live rather than history
  function speedColor(row) {
    var s = root.statsFor(row)
    if (!s)
      return root.dim
    if (s.err !== "" && s.errTs > s.ts)
      return Commons.Color.urgent
    return Date.now() / 1000 - s.ts < 6 ? Commons.Color.accent : root.dim
  }

  // the URL pill is the shortcut to the router's own web UI
  function openRouter() {
    if (!Qt.openUrlExternally(root.baseUrl))
      Util.execArgv(["xdg-open", root.baseUrl])
  }

  function tooltip() {
    if (!root.reachable)
      return "llama router: " + root.statusText
    if (root.active.length === 0)
      return "llama router: no model loaded"
    var parts = []
    for (var i = 0; i < root.active.length; i++) {
      var m = root.active[i]
      parts.push(m.name + " on " + (m.device === "" ? "any card" : m.device)
        + (m.addr === "" ? "" : " at " + m.addr))
    }
    return "llama router: " + parts.join(", ")
  }

  function refresh() {
    modelsProc.running = true
    propsProc.running = true
    stateProc.running = true
    // the speed lines only show in the panel, so the bar does not pay for them
    if (root.opened)
      logsProc.running = true
  }

  // start / stop / restart on the user unit: the router is what Pi talks to, so
  // this is the only place a running conversation can be cut short
  function control(verb) {
    if (ctlProc.running)
      return
    root.actionRunning = true
    root.actionNote = "Sending " + verb + " to " + root.serviceUnit + "..."
    ctlProc.command = ["systemctl", "--user", verb, root.serviceUnit]
    ctlProc.running = true
  }

  // the unit runs in the foreground, so its server log is the journal
  function openLogs() {
    Util.execArgv(["omarchy", "launch", "tui", "--app-id=org.omarchy.llama-router-logs",
      "journalctl", "--user", "-f", "-n", "200", "-u", root.serviceUnit])
    root.close()
  }

  function firstLine(text) {
    var lines = String(text).trim().split("\n")
    return lines.length > 0 ? lines[0] : ""
  }

  function cycleBarMode() {
    var modes = ["device", "count", "name", "off"]
    var next = modes[(modes.indexOf(root.barMode) + 1 + modes.length) % modes.length]
    modeProc.command = ["omarchy", "bar", "set", root.moduleName, "barMode", next]
    modeProc.running = true
  }

  readonly property string barText: {
    if (root.barMode === "off")
      return ""
    if (!root.reachable) {
      if (root.unitState === "active" || root.unitState === "activating")
        return "waking"
      return root.unitState === "failed" ? "failed" : "stopped"
    }
    if (root.barMode === "count")
      return String(root.loaded.length) + (root.busy.length > 0 ? "+" : "")
    if (root.barMode === "name") {
      if (root.active.length === 0)
        return "idle"
      var names = []
      for (var i = 0; i < root.active.length && i < 2; i++)
        names.push(trim(root.active[i].name, 18))
      if (root.active.length > 2)
        names.push("+" + (root.active.length - 2))
      return names.join(" ")
    }
    return root.active.length === 0 ? "idle" : root.cardTags
  }

  Process {
    id: modelsProc
    running: false
    command: ["curl", "-s", "--max-time", "4", root.baseUrl + "/models"]
    stdout: StdioCollector {
      id: modelsOut
      waitForEnd: true
    }
    onExited: function(exitCode) {
      root.applyModels(exitCode, modelsOut.text)
    }
  }

  Process {
    id: propsProc
    running: false
    command: ["curl", "-s", "--max-time", "4", root.baseUrl + "/props"]
    stdout: StdioCollector {
      id: propsOut
      waitForEnd: true
    }
    onExited: function(exitCode) {
      root.applyProps(exitCode, propsOut.text)
    }
  }

  Process {
    id: logsProc
    running: false
    command: ["journalctl", "--user", "--no-pager", "-q", "-o", "short-unix",
      "--grep", root.logFilter, "-n", String(root.logLines), "-u", root.serviceUnit]
    stdout: StdioCollector {
      id: logsOut
      waitForEnd: true
    }
    onExited: function(exitCode) {
      root.applyLogs(exitCode, logsOut.text)
    }
  }

  Process {
    id: modeProc
    running: false
    command: []
  }

  Process {
    id: stateProc
    running: false
    command: ["systemctl", "--user", "is-active", root.serviceUnit]
    stdout: StdioCollector {
      id: stateOut
      waitForEnd: true
    }
    onExited: function(exitCode) {
      var state = String(stateOut.text).trim()
      root.unitState = state === "" ? "unknown" : state
    }
  }

  Process {
    id: ctlProc
    running: false
    command: []
    stdout: StdioCollector {
      id: ctlOut
      waitForEnd: true
    }
    stderr: StdioCollector {
      id: ctlErr
      waitForEnd: true
    }
    onExited: function(exitCode) {
      root.actionRunning = false
      root.actionNote = exitCode === 0 ? "Sent, the router takes a few minutes to load" : firstLine(ctlErr.text)
      // the unit changed, so both the state and the model list are stale now
      root.refresh()
    }
  }

  Timer {
    id: poller
    interval: root.pollSec * 1000
    repeat: true
    running: true
    triggeredOnStart: true
    onTriggered: {
      root.refresh()
      root.updatedAt = Qt.formatTime(new Date(), "HH:mm:ss")
    }
  }

  onOpenedChanged: {
    if (root.opened)
      root.refresh()
  }

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    labelVisible: false
    hasVisualContent: true
    tooltipText: root.tooltip()
    dimmed: !root.reachable
    horizontalMargin: 6
    fixedWidth: root.vertical ? -1 : Math.ceil(barRow.implicitWidth + button.scaledHorizontalMargin * 2)

    onPressed: function(pressedButton) {
      if (pressedButton === Qt.LeftButton)
        root.toggle()
      else if (pressedButton === Qt.RightButton)
        root.cycleBarMode()
      else if (pressedButton === Qt.MiddleButton)
        root.refresh()
    }

    Row {
      id: barRow
      anchors.centerIn: parent
      spacing: Style.space(6)

      Text {
        anchors.verticalCenter: parent.verticalCenter
        text: root.barGlyph
        font.family: root.barGlyphFont
        font.pixelSize: Style.font.icon
        color: root.statusColor
      }

      Text {
        visible: root.barText !== ""
        anchors.verticalCenter: parent.verticalCenter
        text: root.barText
        textFormat: Text.PlainText
        font.family: root.fontFamily
        font.pixelSize: Style.font.body
        color: root.statusColor
        elide: Text.ElideRight
        width: Math.min(implicitWidth, Style.space(260))
      }
    }
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    focusTarget: keyCatcher
    contentWidth: panel.fittedContentWidth(Style.space(400))
    contentHeight: panel.fittedContentHeight(column.implicitHeight, Style.space(520))

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      onCloseRequested: root.close()
      onTabRequested: function(direction) {
        root.switchPanel(direction)
      }
    }

    Column {
      id: column
      width: parent.width
      spacing: Style.space(8)

      PanelHero {
        width: parent.width
        title: "llama router"
        meta: root.statusText
        foreground: root.foreground
        fontFamily: root.fontFamily

        // PanelHero's own detail pill is not clickable, so the URL is its
        // trailing control instead
        trailingControl: Component {
          Button {
            text: root.baseUrl
            foreground: root.foreground
            fontFamily: root.fontFamily
            bordered: true
            tooltipText: "Open " + root.baseUrl + " in the browser"
            onClicked: root.openRouter()
          }
        }
      }

      Row {
        id: actionRow
        width: parent.width
        spacing: Style.space(8)

        readonly property real thirdWidth: (width - spacing * 2) / 3

        Button {
          visible: !root.unitUp
          enabled: !root.actionRunning
          width: actionRow.thirdWidth * 2 + actionRow.spacing
          text: "Start " + root.serviceUnit
          foreground: root.foreground
          fontFamily: root.fontFamily
          onClicked: root.control("start")
        }

        Button {
          visible: root.unitUp
          enabled: !root.actionRunning
          width: actionRow.thirdWidth
          text: "Stop"
          foreground: root.foreground
          fontFamily: root.fontFamily
          onClicked: root.confirmStop = true
        }

        Button {
          visible: root.unitUp
          enabled: !root.actionRunning
          width: actionRow.thirdWidth
          text: "Restart"
          foreground: root.foreground
          fontFamily: root.fontFamily
          onClicked: root.control("restart")
        }

        Button {
          width: actionRow.thirdWidth
          text: "Logs"
          foreground: root.foreground
          fontFamily: root.fontFamily
          tooltipText: "Follow the " + root.serviceUnit + " journal in a new terminal"
          onClicked: root.openLogs()
        }
      }

      Text {
        visible: root.actionNote !== ""
        width: parent.width
        text: root.actionNote
        textFormat: Text.PlainText
        font.family: root.fontFamily
        font.pixelSize: Style.font.caption
        color: root.statusColor
        wrapMode: Text.Wrap
      }

      Repeater {
        model: root.cardGroups

        delegate: Column {
          id: cardBlock
          required property var modelData
          width: parent.width
          spacing: Style.space(4)

          PanelSeparator {
            width: parent.width
            foreground: root.foreground
          }

          PanelSectionHeader {
            width: parent.width
            text: cardBlock.modelData.device === "" ? "ANY CARD" : cardBlock.modelData.device.toUpperCase()
            foreground: root.foreground
            fontFamily: root.fontFamily
          }

          Repeater {
            model: cardBlock.modelData.rows

            delegate: Column {
              id: modelRow
              required property var modelData
              width: parent.width
              spacing: 0

              Row {
                width: parent.width
                spacing: Style.space(8)

                Rectangle {
                  anchors.verticalCenter: parent.verticalCenter
                  width: Style.space(6)
                  height: Style.space(6)
                  radius: height / 2
                  color: modelRow.modelData.status === "loaded" ? Commons.Color.accent
                    : modelRow.modelData.status === "sleeping" ? root.dim : Commons.Color.urgent
                }

                Text {
                  id: nameText
                  anchors.verticalCenter: parent.verticalCenter
                  width: Math.max(Style.space(120), panel.contentWidth - Style.space(210))
                  text: modelRow.modelData.name
                  textFormat: Text.PlainText
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.body
                  color: root.foreground
                  elide: Text.ElideMiddle

                  MouseArea {
                    id: nameHover
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.NoButton
                  }

                  PanelToolTip {
                    // only worth a popup when the name does not fit
                    visible: nameHover.containsMouse && nameText.implicitWidth > nameText.width
                    text: modelRow.modelData.id === modelRow.modelData.name ? modelRow.modelData.name
                      : modelRow.modelData.name + "\n" + modelRow.modelData.id
                    fontFamily: root.fontFamily
                  }
                }

                Text {
                  anchors.verticalCenter: parent.verticalCenter
                  text: modelRow.modelData.status === "loaded" && modelRow.modelData.addr !== "" ? modelRow.modelData.addr : modelRow.modelData.status
                  textFormat: Text.PlainText
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                  color: modelRow.modelData.status === "loaded" ? root.foreground : root.dim
                }
              }

              Text {
                id: speedText
                visible: text !== ""
                width: parent.width
                leftPadding: Style.space(14)
                text: root.speedLine(modelRow.modelData)
                textFormat: Text.PlainText
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
                color: root.speedColor(modelRow.modelData)
                elide: Text.ElideRight

                MouseArea {
                  id: speedHover
                  anchors.fill: parent
                  hoverEnabled: true
                  acceptedButtons: Qt.NoButton
                }

                PanelToolTip {
                  visible: speedHover.containsMouse && speedText.implicitWidth > speedText.width
                  text: speedText.text
                  fontFamily: root.fontFamily
                }
              }
            }
          }
        }
      }

      Column {
        visible: root.showIdle && root.idle.length > 0
        width: parent.width
        spacing: Style.space(4)

        PanelSeparator {
          width: parent.width
          foreground: root.foreground
        }

        PanelSectionHeader {
          width: parent.width
          text: "IDLE (" + root.idle.length + ")"
          foreground: root.foreground
          fontFamily: root.fontFamily
        }

        Text {
          width: parent.width
          text: root.idle.map(function (m) {
            return m.name
          }).join(", ")
          textFormat: Text.PlainText
          font.family: root.fontFamily
          font.pixelSize: Style.font.caption
          color: root.dim
          wrapMode: Text.Wrap
          maximumLineCount: 5
          elide: Text.ElideRight
        }
      }

      PanelSeparator {
        width: parent.width
        foreground: root.foreground
      }

      Text {
        width: parent.width
        text: "unit " + root.serviceUnit + " (" + root.unitState
          + (root.buildInfo === "" ? "" : ", " + root.buildInfo) + ")  -  polled "
          + (root.updatedAt === "" ? "never" : root.updatedAt)
          + "  -  every " + root.pollSec + " s  -  right-click cycles the bar readout"
          + "  -  t/s read from the last " + root.logLines + " journal lines of " + root.serviceUnit
        textFormat: Text.PlainText
        font.family: root.fontFamily
        font.pixelSize: Style.font.caption
        color: root.dim
        wrapMode: Text.Wrap
      }
    }

    ConfirmDialog {
      id: stopConfirm
      anchors.fill: parent
      z: 20
      opened: root.confirmStop
      message: root.loaded.length > 0 ? "Stopping the router drops " + root.loaded.length
        + " loaded model(s) and any conversation using them." : "Stop " + root.serviceUnit + "?"
      cancelText: "Cancel"
      confirmText: "Stop"
      background: Commons.Color.popups.background
      foreground: Commons.Color.popups.text
      scrim: Util.alpha(Commons.Color.background, 0.55)
      fontFamily: root.fontFamily
      onCanceled: root.confirmStop = false
      onConfirmed: {
        root.confirmStop = false
        root.control("stop")
      }
    }
  }
}
