function fail {
  echo $1 >&2
  exit 1
}

# Usage:
# retry ping invalidserver
# https://unix.stackexchange.com/a/137639
function retry {
  local n=1
  local max=99999
  local delay=15
  while true; do
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        ((n++))
        echo "Command failed. Attempt $n/$max starting in $delay seconds..."
        sleep $delay;
      else
        fail "The command has failed after $n attempts."
      fi
    }
  done
}
